"""
Compute Metrics for MATLAB-SPINN Hybrid
"""
import os
import json
import numpy as np
import pandas as pd

def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def compute_metrics(true_path, pred_path, out_path):
    df_true = pd.read_csv(true_path)
    df_pred = pd.read_csv(pred_path)

    metrics = {}
    total_rmse = 0

    for col in ["N", "H", "I"]:
        y_true = df_true[col].values
        y_pred = df_pred[f"{col}_pred"].values

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = r2_score_manual(y_true, y_pred)
        
        metrics[col] = {
            "RMSE": float(rmse),
            "R2": float(r2)
        }
        total_rmse += rmse

    metrics["Global"] = {
        "Mean_RMSE": float(total_rmse / 3.0)
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compute_metrics(
        os.path.join(project_root, "data", "matlab_sode_trajectories.csv"),
        os.path.join(project_root, "data", "spinn_predictions.csv"),
        os.path.join(project_root, "evaluation", "hybrid_metrics_report.json")
    )
