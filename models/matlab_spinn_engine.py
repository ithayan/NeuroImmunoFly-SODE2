"""
Stochastic Physics-Informed Neural Network (S-PINN) Engine (MATLAB Hybrid Version)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SODE_PARAMS = {
    "alpha":    1.0,
    "gamma":    0.3,
    "k_1":      0.5,
    "delta_1":  0.1,
    "k_2":      0.4,
    "delta_2":  0.15,
    "mu":       0.05,
}

T_STRESS = 20.0
EPOCHS = 5000
LR = 1e-3
LAMBDA_PHYS = 1.0
DEVICE = "cpu"

class SPINN(nn.Module):
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

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)

def stress_function_tensor(t, t_onset=T_STRESS):
    return torch.sigmoid(50.0 * (t - t_onset))

def compute_physics_residual(model, t_colloc, params):
    t_colloc.requires_grad_(True)
    X_hat = model(t_colloc)

    N_hat = X_hat[:, 0:1]
    H_hat = X_hat[:, 1:2]
    I_hat = X_hat[:, 2:3]

    dN_dt = torch.autograd.grad(N_hat, t_colloc, grad_outputs=torch.ones_like(N_hat), create_graph=True)[0]
    dH_dt = torch.autograd.grad(H_hat, t_colloc, grad_outputs=torch.ones_like(H_hat), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I_hat, t_colloc, grad_outputs=torch.ones_like(I_hat), create_graph=True)[0]

    S_t = stress_function_tensor(t_colloc)

    res_N = dN_dt - (params["alpha"] - params["gamma"] * N_hat * S_t)
    res_H = dH_dt - (params["k_1"] * N_hat - params["delta_1"] * H_hat)
    res_I = dI_dt - (params["k_2"] * H_hat - params["delta_2"] * I_hat - params["mu"] * I_hat**2)

    return torch.mean(res_N**2 + res_H**2 + res_I**2)

def train_spinn(data_path, model_path, pred_path):
    df = pd.read_csv(data_path)
    t_data = df["t"].values.astype(np.float32)
    N_data = df["N"].values.astype(np.float32)
    H_data = df["H"].values.astype(np.float32)
    I_data = df["I"].values.astype(np.float32)

    t_max = max(t_data.max(), 1.0)
    N_max = max(N_data.max(), 1.0)
    H_max = max(H_data.max(), 1.0)
    I_max = max(I_data.max(), 1.0)

    t_norm = t_data / t_max
    N_norm = N_data / N_max
    H_norm = H_data / H_max
    I_norm = I_data / I_max

    t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_target = torch.stack([
        torch.tensor(N_norm, dtype=torch.float32),
        torch.tensor(H_norm, dtype=torch.float32),
        torch.tensor(I_norm, dtype=torch.float32),
    ], dim=1).to(DEVICE)

    t_colloc = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    norm_params = {
        "alpha":    SODE_PARAMS["alpha"] * t_max / N_max,
        "gamma":    SODE_PARAMS["gamma"] * N_max * t_max / N_max,
        "k_1":      SODE_PARAMS["k_1"] * N_max * t_max / H_max,
        "delta_1":  SODE_PARAMS["delta_1"] * t_max,
        "k_2":      SODE_PARAMS["k_2"] * H_max * t_max / I_max,
        "delta_2":  SODE_PARAMS["delta_2"] * t_max,
        "mu":       SODE_PARAMS["mu"] * I_max * t_max,
    }

    global T_STRESS
    t_stress_norm = T_STRESS / t_max

    model = SPINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_fn = nn.MSELoss()

    print(f"Training S-PINN on MATLAB trajectories for {EPOCHS} epochs...")
    
    # Optional: ensure weights dir exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        X_pred = model(t_tensor)
        loss_data = mse_fn(X_pred, X_target)

        _old_stress = T_STRESS
        T_STRESS = t_stress_norm
        loss_phys = compute_physics_residual(model, t_colloc.clone(), norm_params)
        T_STRESS = _old_stress

        loss_total = loss_data + LAMBDA_PHYS * loss_phys
        loss_total.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:4d}/{EPOCHS} | L_total={loss_total.item():.4f}")

    torch.save(model.state_dict(), model_path)
    
    model.eval()
    with torch.no_grad():
        X_pred_final = model(t_tensor).cpu().numpy()

    pred_df = pd.DataFrame({
        "t": t_data,
        "N_pred": X_pred_final[:, 0] * N_max,
        "H_pred": X_pred_final[:, 1] * H_max,
        "I_pred": X_pred_final[:, 2] * I_max,
    })
    pred_df.to_csv(pred_path, index=False)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_spinn(
        os.path.join(project_root, "data", "matlab_sode_trajectories.csv"),
        os.path.join(project_root, "outputs", "weights", "matlab_spinn_model.pth"),
        os.path.join(project_root, "data", "spinn_predictions.csv")
    )
