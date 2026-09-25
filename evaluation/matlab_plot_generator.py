"""
Generate Publication-Grade Figures for MATLAB-SPINN Hybrid
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots(true_path, pred_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    df_true = pd.read_csv(true_path)
    df_pred = pd.read_csv(pred_path)
    
    t = df_true["t"].values
    
    # Plot 1: Trajectory Cascade
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, col in enumerate(["N", "H", "I"]):
        axs[i].plot(t, df_true[col], label=f"MATLAB {col}", color="black", linestyle="--")
        axs[i].plot(t, df_pred[f"{col}_pred"], label=f"S-PINN {col}_hat", color="blue", alpha=0.7)
        axs[i].axvline(20.0, color="red", linestyle=":", label="Stress Onset")
        axs[i].set_ylabel(col)
        axs[i].legend(loc="upper right")
    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "matlab_vs_pinn_cascade.png"), dpi=300)
    plt.close()
    
    # Plot 2: Phase Portrait
    plt.figure(figsize=(6, 5))
    plt.plot(df_true["H"], df_true["I"], color="black", alpha=0.5, label="MATLAB Ground Truth")
    plt.plot(df_pred["H_pred"], df_pred["I_pred"], color="blue", alpha=0.5, label="S-PINN Prediction")
    plt.xlabel("Hormonal Compartment (H)")
    plt.ylabel("Immune Compartment (I)")
    plt.title("Phase Portrait Attractor")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase_portrait_attractor.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_plots(
        os.path.join(project_root, "data", "matlab_sode_trajectories.csv"),
        os.path.join(project_root, "data", "spinn_predictions.csv"),
        os.path.join(project_root, "outputs", "figures")
    )
    print("Plots generated successfully in outputs/figures")
