"""
Stochastic Physics-Informed Neural Network (S-PINN) Engine
==========================================================
A PyTorch-based MLP that learns the SODE trajectories while being
constrained by the underlying physics (governing differential equations)
through an autograd-computed residual loss.

Architecture:
    Input  : 1  (time t)
    Hidden : 4 x 64 with Tanh activation
    Output : 3  [N_hat, H_hat, I_hat]

Loss Function:
    L_total = L_MSE_data + lambda_phys * L_physics_residual

where L_physics_residual penalizes violations of:
    dN/dt = alpha - gamma * N * S(t)
    dH/dt = k_1 * N - delta_1 * H
    dI/dt = k_2 * H - delta_2 * I - mu * I^2

Outputs:
    outputs/weights/spinn_model.pth  --  trained model state dict
    data/spinn_predictions.csv       --  predicted trajectories
    data/loss_history.npy            --  training loss logs
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------- SODE Parameters (must match sode_simulator.py) ----------
SODE_PARAMS = {
    "alpha":    1.0,
    "gamma":    0.3,
    "k_1":      0.5,
    "delta_1":  0.1,
    "k_2":      0.4,
    "delta_2":  0.15,
    "mu":       0.05,
}

T_STRESS = 20.0  # stress onset time

# Training hyperparameters
EPOCHS       = 5000
LR           = 1e-3
LAMBDA_PHYS  = 1.0    # physics loss weighting
BATCH_SIZE   = None    # full-batch training
DEVICE       = "cpu"   # use CPU for portability


class SPINN(nn.Module):
    """
    Stochastic Physics-Informed Neural Network.
    MLP: 1 -> 64 -> 64 -> 64 -> 64 -> 3
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
        )

        # Initialize weights (Xavier)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        """
        Parameters
        ----------
        t : Tensor, shape (batch, 1)
            Normalized time input

        Returns
        -------
        X_hat : Tensor, shape (batch, 3)
            Predicted [N_hat, H_hat, I_hat]
        """
        return self.net(t)


def stress_function_tensor(t, t_onset=T_STRESS):
    """Differentiable approximation of step function using sigmoid."""
    # Smooth approximation for autograd compatibility
    return torch.sigmoid(50.0 * (t - t_onset))


def compute_physics_residual(model, t_colloc, params):
    """
    Compute the physics residual by differentiating the network
    output w.r.t. time using torch.autograd.grad.

    Parameters
    ----------
    model : SPINN
        The neural network
    t_colloc : Tensor, shape (n, 1), requires_grad=True
        Collocation time points
    params : dict
        SODE parameters

    Returns
    -------
    residual : Tensor, scalar
        Mean squared physics residual
    """
    t_colloc.requires_grad_(True)
    X_hat = model(t_colloc)

    N_hat = X_hat[:, 0:1]
    H_hat = X_hat[:, 1:2]
    I_hat = X_hat[:, 2:3]

    # Compute time derivatives via autograd
    dN_dt = torch.autograd.grad(
        N_hat, t_colloc, grad_outputs=torch.ones_like(N_hat),
        create_graph=True, retain_graph=True
    )[0]

    dH_dt = torch.autograd.grad(
        H_hat, t_colloc, grad_outputs=torch.ones_like(H_hat),
        create_graph=True, retain_graph=True
    )[0]

    dI_dt = torch.autograd.grad(
        I_hat, t_colloc, grad_outputs=torch.ones_like(I_hat),
        create_graph=True, retain_graph=True
    )[0]

    # Stress function
    S_t = stress_function_tensor(t_colloc)

    # Physics equations (residuals should be zero if perfectly learned)
    res_N = dN_dt - (params["alpha"] - params["gamma"] * N_hat * S_t)
    res_H = dH_dt - (params["k_1"] * N_hat - params["delta_1"] * H_hat)
    res_I = dI_dt - (params["k_2"] * H_hat - params["delta_2"] * I_hat - params["mu"] * I_hat**2)

    # Mean squared residual
    residual = torch.mean(res_N**2 + res_H**2 + res_I**2)
    return residual


def train_spinn(data_path: str, model_path: str, pred_path: str,
                loss_path: str, epochs: int = EPOCHS):
    """
    Train the S-PINN model.

    Parameters
    ----------
    data_path : str
        Path to sode_trajectories.csv
    model_path : str
        Path to save trained model weights
    pred_path : str
        Path to save predictions CSV
    loss_path : str
        Path to save loss history
    epochs : int
        Number of training epochs
    """
    # --- Load reference data ---
    df = pd.read_csv(data_path)
    t_data = df["time"].values.astype(np.float32)
    N_data = df["N"].values.astype(np.float32)
    H_data = df["H"].values.astype(np.float32)
    I_data = df["I"].values.astype(np.float32)

    # Normalization for numerical stability
    t_max = t_data.max()
    N_max = max(N_data.max(), 1.0)
    H_max = max(H_data.max(), 1.0)
    I_max = max(I_data.max(), 1.0)

    t_norm = t_data / t_max
    N_norm = N_data / N_max
    H_norm = H_data / H_max
    I_norm = I_data / I_max

    # Convert to tensors
    t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_target = torch.stack([
        torch.tensor(N_norm, dtype=torch.float32),
        torch.tensor(H_norm, dtype=torch.float32),
        torch.tensor(I_norm, dtype=torch.float32),
    ], dim=1).to(DEVICE)

    # Collocation points (include training points + extra)
    t_colloc = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # Normalized SODE parameters (adjust for normalization)
    norm_params = {
        "alpha":    SODE_PARAMS["alpha"] * t_max / N_max,
        "gamma":    SODE_PARAMS["gamma"] * N_max * t_max / N_max,
        "k_1":      SODE_PARAMS["k_1"] * N_max * t_max / H_max,
        "delta_1":  SODE_PARAMS["delta_1"] * t_max,
        "k_2":      SODE_PARAMS["k_2"] * H_max * t_max / I_max,
        "delta_2":  SODE_PARAMS["delta_2"] * t_max,
        "mu":       SODE_PARAMS["mu"] * I_max * t_max,
    }

    # Adjust stress onset for normalized time
    global T_STRESS
    t_stress_norm = T_STRESS / t_max

    # --- Model, optimizer ---
    model = SPINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Loss history ---
    loss_history = {
        "epoch": [],
        "loss_total": [],
        "loss_data": [],
        "loss_physics": [],
    }

    mse_fn = nn.MSELoss()

    print(f"  Training S-PINN for {epochs} epochs on {DEVICE}...")
    print(f"  Data points    : {len(t_data)}")
    print(f"  Lambda (phys)  : {LAMBDA_PHYS}")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        X_pred = model(t_tensor)

        # Data loss
        loss_data = mse_fn(X_pred, X_target)

        # Physics residual loss
        # Use smooth stress function with normalized onset
        _old_stress = T_STRESS
        T_STRESS = t_stress_norm  # temporarily set for stress function
        loss_phys = compute_physics_residual(model, t_colloc.clone(), norm_params)
        T_STRESS = _old_stress

        # Total loss
        loss_total = loss_data + LAMBDA_PHYS * loss_phys

        # Backward + optimize
        loss_total.backward()
        optimizer.step()

        # Log
        loss_history["epoch"].append(epoch)
        loss_history["loss_total"].append(loss_total.item())
        loss_history["loss_data"].append(loss_data.item())
        loss_history["loss_physics"].append(loss_phys.item())

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:5d}/{epochs}  |  "
                  f"L_total={loss_total.item():.6f}  "
                  f"L_data={loss_data.item():.6f}  "
                  f"L_phys={loss_phys.item():.6f}")

    # --- Save model ---
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved -> {model_path}")

    # --- Save predictions ---
    model.eval()
    with torch.no_grad():
        X_pred_final = model(t_tensor).cpu().numpy()

    # Denormalize predictions
    pred_df = pd.DataFrame({
        "time":   t_data,
        "N_pred": X_pred_final[:, 0] * N_max,
        "H_pred": X_pred_final[:, 1] * H_max,
        "I_pred": X_pred_final[:, 2] * I_max,
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved -> {pred_path}")

    # --- Save loss history ---
    np.save(loss_path, loss_history)
    print(f"  Loss history saved -> {loss_path}")

    return model, loss_history


def run(project_root: str = None):
    """Main entry point for S-PINN training."""
    if project_root is None:
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    data_path  = os.path.join(project_root, "data", "sode_trajectories.csv")
    model_path = os.path.join(project_root, "outputs", "weights", "spinn_model.pth")
    pred_path  = os.path.join(project_root, "data", "spinn_predictions.csv")
    loss_path  = os.path.join(project_root, "data", "loss_history.npy")

    print("[Phase 3] Training S-PINN...")
    model, history = train_spinn(data_path, model_path, pred_path, loss_path)
    print("[Phase 3] Complete.\n")

    return model, history


if __name__ == "__main__":
    run()
