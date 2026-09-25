"""
Extended S-PINN for the 7-Variable Neuro-Immuno-Endocrine System
=================================================================
Physics-Informed Neural Network constrained by the extended SODE:

  State: [N, H, I, C, F, G1, G2]

Architectural upgrades over v1:
  - Wider network (128 units) for 7 output dimensions
  - 5 hidden layers for increased representational capacity
  - Adaptive physics weight scheduling (curriculum learning)
  - Separate collocation grids for fast/slow timescale equations
  - Smooth sigmoid stress approximation for autograd

Architecture:
  Input: 1 (time t)
  Hidden: 5 x 128 with Tanh
  Output: 7 [N, H, I, C, F, G1, G2]

Loss:
  L_total = L_data + lambda(epoch) * L_physics
  lambda(epoch) = lambda_0 * (1 - exp(-epoch / tau_lambda))

Outputs:
  outputs/weights/spinn_model_v2.pth  -- trained model
  data/spinn_predictions_v2.csv       -- predicted trajectories
  data/loss_history_v2.npy            -- training loss logs
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.sode_extended import PARAMS_V2, STATE_NAMES, T_STRESS

# Training hyperparameters
EPOCHS      = 8000
LR          = 5e-4
LAMBDA_0    = 1.5       # peak physics weight
TAU_LAMBDA  = 1500.0    # curriculum ramp-up timescale
DEVICE      = "cpu"


class SPINNExtended(nn.Module):
    """
    Extended S-PINN: 1 -> 128x5 -> 7
    """

    def __init__(self, n_hidden=128, n_layers=5, n_out=7):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, n_out))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


def stress_smooth(t, t_onset, steepness=50.0):
    return torch.sigmoid(steepness * (t - t_onset))


def circadian_output_tensor(G1, mu_c):
    amp = torch.sqrt(torch.clamp(torch.tensor(mu_c), min=1e-8))
    return torch.clamp((1.0 + G1 / amp) / 2.0, 0.0, 1.0)


def compute_extended_physics_residual(model, t_col, params, t_stress_norm):
    """
    Autograd-based physics residuals for all 7 equations.
    """
    t_col = t_col.requires_grad_(True)
    X = model(t_col)

    N  = X[:, 0:1]
    H  = X[:, 1:2]
    I  = X[:, 2:3]
    C  = X[:, 3:4]
    F  = X[:, 4:5]
    G1 = X[:, 5:6]
    G2 = X[:, 6:7]

    ones = torch.ones_like(N)

    # Time derivatives via autograd
    derivs = []
    for j in range(7):
        d = torch.autograd.grad(
            X[:, j:j+1], t_col,
            grad_outputs=ones,
            create_graph=True, retain_graph=True
        )[0]
        derivs.append(d)

    dN, dH, dI, dC, dF, dG1, dG2 = derivs

    # Auxiliary quantities
    S = stress_smooth(t_col, t_stress_norm)
    mu_c = params["mu_c"]
    omega = 2.0 * np.pi / params["T_circ"]
    Phi = circadian_output_tensor(G1, mu_c)

    eps = 1e-8

    # Residuals
    r_sq = G1**2 + G2**2
    res_G1 = dG1 - (mu_c * G1 - omega * G2 - G1 * r_sq)
    res_G2 = dG2 - (omega * G1 + mu_c * G2 - G2 * r_sq)

    res_N = dN - (
        params["alpha"] * Phi
        - params["gamma"] * N * S
        - params["kappa_f"] * F * N / (params["K_N"] + N + eps)
    )

    res_H = dH - (
        params["k1"] * N * Phi
        - params["delta1"] * H
        - params["eta"] * C * H / (params["K_H"] + H + eps)
    )

    hill = params["beta_I"] * I**2 / (params["K_b"]**2 + I**2 + eps)
    res_I = dI - (
        params["k2"] * H
        + hill
        - params["delta2"] * I
        - params["mu"] * I**2
    )

    res_C = dC - (
        params["k3"] * I * (1.0 + params["lambda_c"] * (1.0 - Phi))
        - params["delta3"] * C
    )

    res_F = dF - (
        params["k4"] * H
        - params["delta4"] * F
        - params["psi"] * F**2
    )

    total_res = (
        torch.mean(res_N**2) +
        torch.mean(res_H**2) +
        torch.mean(res_I**2) +
        torch.mean(res_C**2) +
        torch.mean(res_F**2) +
        torch.mean(res_G1**2) +
        torch.mean(res_G2**2)
    )

    return total_res


def adaptive_lambda(epoch, lambda_0=LAMBDA_0, tau=TAU_LAMBDA):
    """Curriculum learning: ramp physics weight from 0 to lambda_0."""
    return lambda_0 * (1.0 - np.exp(-epoch / tau))


def train_spinn_extended(data_path, model_path, pred_path, loss_path,
                         epochs=EPOCHS):
    df = pd.read_csv(data_path)
    t_data = df["time"].values.astype(np.float32)

    # All 7 state columns
    col_names = STATE_NAMES
    state_data = {}
    for name in col_names:
        state_data[name] = df[name].values.astype(np.float32)

    # Normalization
    t_max = t_data.max()
    t_norm = t_data / t_max

    maxvals = {}
    normed = {}
    for name in col_names:
        mx = max(np.abs(state_data[name]).max(), 1e-6)
        maxvals[name] = mx
        normed[name] = state_data[name] / mx

    t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    target_cols = []
    for name in col_names:
        target_cols.append(torch.tensor(normed[name], dtype=torch.float32))
    X_target = torch.stack(target_cols, dim=1).to(DEVICE)

    t_colloc = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # Normalize SODE parameters for normalized time/state domain
    p = PARAMS_V2
    norm_params = {
        "alpha":    p["alpha"] * t_max / maxvals["N"],
        "gamma":    p["gamma"] * t_max,
        "kappa_f":  p["kappa_f"] * maxvals["F"] * t_max / maxvals["N"],
        "K_N":      p["K_N"] / maxvals["N"],
        "k1":       p["k1"] * maxvals["N"] * t_max / maxvals["H"],
        "delta1":   p["delta1"] * t_max,
        "eta":      p["eta"] * maxvals["C"] * t_max / maxvals["H"],
        "K_H":      p["K_H"] / maxvals["H"],
        "k2":       p["k2"] * maxvals["H"] * t_max / maxvals["I"],
        "beta_I":   p["beta_I"] * t_max / maxvals["I"],
        "K_b":      p["K_b"] / maxvals["I"],
        "delta2":   p["delta2"] * t_max,
        "mu":       p["mu"] * maxvals["I"] * t_max,
        "k3":       p["k3"] * maxvals["I"] * t_max / maxvals["C"],
        "lambda_c": p["lambda_c"],
        "delta3":   p["delta3"] * t_max,
        "k4":       p["k4"] * maxvals["H"] * t_max / maxvals["F"],
        "delta4":   p["delta4"] * t_max,
        "psi":      p["psi"] * maxvals["F"] * t_max,
        "mu_c":     p["mu_c"],
        "T_circ":   p["T_circ"] / t_max,
    }

    t_stress_norm = T_STRESS / t_max

    model = SPINNExtended().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    loss_history = {"epoch": [], "loss_total": [], "loss_data": [],
                    "loss_physics": [], "lambda_phys": []}
    mse_fn = nn.MSELoss()

    print(f"  Training extended S-PINN for {epochs} epochs on {DEVICE}...")
    print(f"  Architecture: 1 -> 128x5 -> 7")
    print(f"  Data points: {len(t_data)}")
    print(f"  Curriculum lambda: 0 -> {LAMBDA_0} (tau={TAU_LAMBDA})")

    for epoch in range(epochs):
        optimizer.zero_grad()

        X_pred = model(t_tensor)
        loss_data = mse_fn(X_pred, X_target)

        lam = adaptive_lambda(epoch)
        loss_phys = compute_extended_physics_residual(
            model, t_colloc.clone(), norm_params, t_stress_norm
        )

        loss_total = loss_data + lam * loss_phys
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        loss_history["epoch"].append(epoch)
        loss_history["loss_total"].append(loss_total.item())
        loss_history["loss_data"].append(loss_data.item())
        loss_history["loss_physics"].append(loss_phys.item())
        loss_history["lambda_phys"].append(lam)

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:5d}/{epochs}  |  "
                  f"L_total={loss_total.item():.6f}  "
                  f"L_data={loss_data.item():.6f}  "
                  f"L_phys={loss_phys.item():.6f}  "
                  f"lambda={lam:.3f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved -> {model_path}")

    # Save predictions (denormalized)
    model.eval()
    with torch.no_grad():
        X_pred_final = model(t_tensor).cpu().numpy()

    pred_df = pd.DataFrame({"time": t_data})
    for j, name in enumerate(col_names):
        pred_df[f"{name}_pred"] = X_pred_final[:, j] * maxvals[name]
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved -> {pred_path}")

    np.save(loss_path, loss_history)
    print(f"  Loss history saved -> {loss_path}")

    return model, loss_history


def run(project_root: str = None):
    if project_root is None:
        project_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."
        )

    data_path  = os.path.join(project_root, "data", "sode_trajectories_v2.csv")
    model_path = os.path.join(project_root, "outputs", "weights",
                               "spinn_model_v2.pth")
    pred_path  = os.path.join(project_root, "data", "spinn_predictions_v2.csv")
    loss_path  = os.path.join(project_root, "data", "loss_history_v2.npy")

    print("[Phase 3 v2] Training extended S-PINN (7 variables)...")
    model, history = train_spinn_extended(
        data_path, model_path, pred_path, loss_path
    )
    print("[Phase 3 v2] Complete.\n")
    return model, history


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    run()
