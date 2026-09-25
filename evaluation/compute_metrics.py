"""
Metrics Computation for NeuroImmunoFly-SODE Pipeline
=====================================================
Computes quantitative evaluation metrics comparing SODE ground-truth
trajectories against S-PINN predictions.

Metrics:
  - RMSE per compartment (N, H, I)
  - Maximum physics violation (residual error)
  - Mean Absolute Error (MAE) per compartment
  - R^2 coefficient of determination

Output:
  evaluation/metrics_report.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

from models.sode_simulator import PARAMS as SODE_PARAMS, stress_function


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_physics_residuals(t, N, H, I, params):
    """
    Compute the instantaneous physics residuals for each time point
    using finite differences for the time derivatives.

    Parameters
    ----------
    t : ndarray
        Time array
    N, H, I : ndarray
        Predicted compartment trajectories
    params : dict
        SODE parameters

    Returns
    -------
    residuals : dict
        Per-equation residuals and max violation
    """
    dt = t[1] - t[0]

    # Numerical derivatives (central difference, forward/backward at boundaries)
    dN_dt = np.gradient(N, dt)
    dH_dt = np.gradient(H, dt)
    dI_dt = np.gradient(I, dt)

    # Stress function
    S = np.array([stress_function(ti) for ti in t])

    # Expected derivatives from SODE
    dN_expected = params["alpha"] - params["gamma"] * N * S
    dH_expected = params["k_1"] * N - params["delta_1"] * H
    dI_expected = params["k_2"] * H - params["delta_2"] * I - params["mu"] * I**2

    # Residuals
    res_N = np.abs(dN_dt - dN_expected)
    res_H = np.abs(dH_dt - dH_expected)
    res_I = np.abs(dI_dt - dI_expected)

    return {
        "mean_residual_N": float(np.mean(res_N)),
        "mean_residual_H": float(np.mean(res_H)),
        "mean_residual_I": float(np.mean(res_I)),
        "max_violation_N": float(np.max(res_N)),
        "max_violation_H": float(np.max(res_H)),
        "max_violation_I": float(np.max(res_I)),
        "max_total_violation": float(np.max(res_N + res_H + res_I)),
    }


def run(project_root: str = None):
    """Main entry point for metrics computation."""
    if project_root is None:
        project_root = PROJECT_ROOT

    traj_path = os.path.join(project_root, "data", "sode_trajectories.csv")
    pred_path = os.path.join(project_root, "data", "spinn_predictions.csv")
    out_path  = os.path.join(project_root, "evaluation", "metrics_report.json")

    print("[Phase 4a] Computing evaluation metrics...")

    # Load data
    df_true = pd.read_csv(traj_path)
    df_pred = pd.read_csv(pred_path)

    t = df_true["time"].values
    N_true = df_true["N"].values
    H_true = df_true["H"].values
    I_true = df_true["I"].values

    N_pred = df_pred["N_pred"].values
    H_pred = df_pred["H_pred"].values
    I_pred = df_pred["I_pred"].values

    # --- Per-compartment metrics ---
    metrics = {
        "compartment_metrics": {
            "N": {
                "RMSE": compute_rmse(N_true, N_pred),
                "MAE":  compute_mae(N_true, N_pred),
                "R2":   compute_r2(N_true, N_pred),
            },
            "H": {
                "RMSE": compute_rmse(H_true, H_pred),
                "MAE":  compute_mae(H_true, H_pred),
                "R2":   compute_r2(H_true, H_pred),
            },
            "I": {
                "RMSE": compute_rmse(I_true, I_pred),
                "MAE":  compute_mae(I_true, I_pred),
                "R2":   compute_r2(I_true, I_pred),
            },
        },
        "physics_violation": compute_physics_residuals(
            t, N_pred, H_pred, I_pred, SODE_PARAMS
        ),
        "global_RMSE": compute_rmse(
            np.column_stack([N_true, H_true, I_true]),
            np.column_stack([N_pred, H_pred, I_pred]),
        ),
        "data_points": len(t),
        "time_range": [float(t.min()), float(t.max())],
    }

    # Save report
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"  RMSE(N) = {metrics['compartment_metrics']['N']['RMSE']:.6f}")
    print(f"  RMSE(H) = {metrics['compartment_metrics']['H']['RMSE']:.6f}")
    print(f"  RMSE(I) = {metrics['compartment_metrics']['I']['RMSE']:.6f}")
    print(f"  R2(N)   = {metrics['compartment_metrics']['N']['R2']:.6f}")
    print(f"  R2(H)   = {metrics['compartment_metrics']['H']['R2']:.6f}")
    print(f"  R2(I)   = {metrics['compartment_metrics']['I']['R2']:.6f}")
    print(f"  Max physics violation = {metrics['physics_violation']['max_total_violation']:.6f}")
    print(f"  Report saved -> {out_path}")
    print("[Phase 4a] Complete.\n")

    return metrics


if __name__ == "__main__":
    run()
