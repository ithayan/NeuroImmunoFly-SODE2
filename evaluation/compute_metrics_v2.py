"""
Extended Metrics for the 7-Variable NeuroImmunoFly-SODE v2
============================================================
Computes quantitative evaluation for all compartments including:
  - Per-compartment RMSE, MAE, R^2
  - Physics violation residuals for all 7 equations
  - Circadian period estimation via FFT
  - Bistability detection via hysteresis analysis
  - Feedback loop strength metrics

Output:
  evaluation/metrics_report_v2.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

from models.sode_extended import (
    PARAMS_V2, STATE_NAMES, stress_function, circadian_output,
    sode_extended_drift,
)


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_extended_physics_residuals(t, X_pred, params):
    """
    Finite-difference physics residuals for all 7 equations.
    """
    dt = t[1] - t[0]
    n = len(t)

    derivs_pred = np.zeros((n, 7))
    for j in range(7):
        derivs_pred[:, j] = np.gradient(X_pred[:, j], dt)

    residuals = {}
    for j, name in enumerate(STATE_NAMES):
        derivs_expected = np.zeros(n)
        for i in range(n):
            drift = sode_extended_drift(X_pred[i], t[i], params)
            derivs_expected[i] = drift[j]

        res = np.abs(derivs_pred[:, j] - derivs_expected)
        residuals[name] = {
            "mean_residual": float(np.mean(res)),
            "max_violation": float(np.max(res)),
            "median_residual": float(np.median(res)),
        }

    # Total violation
    total = sum(
        residuals[name]["mean_residual"] for name in STATE_NAMES
    )
    residuals["total_mean_violation"] = float(total)

    return residuals


def estimate_circadian_period(t, G1, expected_period=24.0):
    """
    Estimate circadian period from G1 oscillator via FFT.
    """
    dt = t[1] - t[0]
    n = len(G1)

    # Detrend
    G1_detrended = G1 - np.mean(G1)

    # FFT
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_mag = np.abs(np.fft.rfft(G1_detrended))

    # Find dominant frequency (exclude DC)
    fft_mag[0] = 0
    if len(fft_mag) < 2:
        return {"estimated_period": float("nan"), "error_pct": float("nan")}

    dominant_idx = np.argmax(fft_mag)
    dominant_freq = freqs[dominant_idx]

    if dominant_freq == 0:
        return {"estimated_period": float("nan"), "error_pct": float("nan")}

    estimated_period = 1.0 / dominant_freq
    error_pct = abs(estimated_period - expected_period) / expected_period * 100

    return {
        "estimated_period": float(estimated_period),
        "expected_period": float(expected_period),
        "error_pct": float(error_pct),
        "dominant_frequency": float(dominant_freq),
    }


def detect_bistability(I_trajectory, threshold_low=0.5, threshold_high=2.0):
    """
    Detect bistability in immune compartment by analyzing the
    distribution of steady-state values.

    A bistable system will show a bimodal distribution with values
    clustered around two distinct attractors.
    """
    # Use second half of trajectory (transient-free)
    I_ss = I_trajectory[len(I_trajectory) // 2:]

    histogram, bin_edges = np.histogram(I_ss, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks (local maxima in histogram)
    peaks = []
    for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            if histogram[i] > len(I_ss) * 0.02:
                peaks.append(bin_centers[i])

    is_bistable = len(peaks) >= 2

    return {
        "is_bistable": bool(is_bistable),
        "n_peaks": len(peaks),
        "peak_values": [float(p) for p in peaks],
        "I_mean": float(np.mean(I_ss)),
        "I_std": float(np.std(I_ss)),
        "I_range": [float(I_ss.min()), float(I_ss.max())],
    }


def compute_feedback_strength(t, X_trajectories, params):
    """
    Quantify feedback loop strength by measuring the correlation
    between upstream cause and downstream effect with appropriate lags.
    """
    N = X_trajectories[:, 0]
    H = X_trajectories[:, 1]
    I = X_trajectories[:, 2]
    C = X_trajectories[:, 3]
    F = X_trajectories[:, 4]

    # Feedforward: N -> H -> I (should be positively correlated)
    corr_NH = float(np.corrcoef(N[:-10], H[10:])[0, 1])
    corr_HI = float(np.corrcoef(H[:-10], I[10:])[0, 1])

    # Feedback: F -> N (negative feedback, should be negatively correlated)
    corr_FN = float(np.corrcoef(F[:-20], N[20:])[0, 1])

    # Cytokine feedback: C -> H (negative modulation)
    corr_CH = float(np.corrcoef(C[:-10], H[10:])[0, 1])

    return {
        "feedforward_N_H": corr_NH,
        "feedforward_H_I": corr_HI,
        "feedback_F_N": corr_FN,
        "feedback_C_H": corr_CH,
        "negative_feedback_intact": corr_FN < 0,
    }


def run(project_root: str = None):
    if project_root is None:
        project_root = PROJECT_ROOT

    traj_path = os.path.join(project_root, "data", "sode_trajectories_v2.csv")
    pred_path = os.path.join(project_root, "data", "spinn_predictions_v2.csv")
    out_path  = os.path.join(project_root, "evaluation",
                              "metrics_report_v2.json")

    print("[Phase 4a v2] Computing extended evaluation metrics...")

    df_true = pd.read_csv(traj_path)
    df_pred = pd.read_csv(pred_path)

    t = df_true["time"].values

    # Per-compartment metrics
    comp_metrics = {}
    for name in STATE_NAMES:
        y_true = df_true[name].values
        y_pred = df_pred[f"{name}_pred"].values
        comp_metrics[name] = {
            "RMSE": compute_rmse(y_true, y_pred),
            "MAE":  compute_mae(y_true, y_pred),
            "R2":   compute_r2(y_true, y_pred),
        }

    # Build predicted state matrix
    X_pred = np.column_stack([
        df_pred[f"{name}_pred"].values for name in STATE_NAMES
    ])

    # Physics residuals
    physics = compute_extended_physics_residuals(t, X_pred, PARAMS_V2)

    # Circadian period
    circadian = estimate_circadian_period(
        t, df_true["G1"].values, expected_period=PARAMS_V2["T_circ"]
    )

    # Bistability
    bistability = detect_bistability(df_true["I"].values)

    # Feedback strength (from true trajectories)
    X_true = np.column_stack([df_true[name].values for name in STATE_NAMES])
    feedback = compute_feedback_strength(t, X_true, PARAMS_V2)

    # Global RMSE (all compartments)
    X_true_all = np.column_stack([df_true[name].values for name in STATE_NAMES])
    X_pred_all = np.column_stack([
        df_pred[f"{name}_pred"].values for name in STATE_NAMES
    ])
    global_rmse = compute_rmse(X_true_all, X_pred_all)

    metrics = {
        "compartment_metrics": comp_metrics,
        "physics_violation": physics,
        "circadian_analysis": circadian,
        "bistability_analysis": bistability,
        "feedback_strength": feedback,
        "global_RMSE": global_rmse,
        "data_points": len(t),
        "time_range": [float(t.min()), float(t.max())],
        "n_state_variables": 7,
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Per-compartment RMSE:")
    for name in STATE_NAMES:
        r = comp_metrics[name]
        print(f"    {name:4s}: RMSE={r['RMSE']:.6f}  R2={r['R2']:.6f}")
    print(f"  Global RMSE: {global_rmse:.6f}")
    print(f"  Circadian period: {circadian.get('estimated_period', 'N/A')}")
    print(f"  Bistability detected: {bistability['is_bistable']}")
    print(f"  Negative feedback intact: {feedback['negative_feedback_intact']}")
    print(f"  Report saved -> {out_path}")
    print("[Phase 4a v2] Complete.\n")

    return metrics


if __name__ == "__main__":
    run()
