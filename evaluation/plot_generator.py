"""
Publication-Grade Figure Generator for NeuroImmunoFly-SODE
===========================================================
Generates three 300 DPI figures for manuscript/report inclusion.

Figures:
  1. trajectory_cascade.png  -- 3-panel true vs. predicted time-series
  2. phase_portrait.png      -- H(t) vs I(t) phase space trajectory
  3. loss_convergence.png    -- log-scale training loss curves

Output directory: outputs/figures/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Color palette (publication-friendly, colorblind-safe)
COLORS = {
    "true":  "#2c3e50",    # dark navy
    "pred":  "#e74c3c",    # scientific red
    "N":     "#3498db",    # blue
    "H":     "#e67e22",    # orange
    "I":     "#27ae60",    # green
    "data":  "#9b59b6",    # purple
    "phys":  "#e74c3c",    # red
}


def plot_trajectory_cascade(df_true, df_pred, output_path):
    """
    3-panel subplot comparing true (SODE) vs predicted (S-PINN)
    time-series for N(t), H(t), I(t).
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    compartments = [
        ("N", "N_pred", r"$N(t)$ -- Neuronal Compartment",   COLORS["N"]),
        ("H", "H_pred", r"$H(t)$ -- Hormonal Compartment",   COLORS["H"]),
        ("I", "I_pred", r"$I(t)$ -- Immune Compartment",     COLORS["I"]),
    ]

    t = df_true["time"].values

    for ax, (true_col, pred_col, title, color) in zip(axes, compartments):
        # True trajectory
        ax.plot(t, df_true[true_col], color=COLORS["true"], linewidth=1.8,
                label="SODE (ground truth)", alpha=0.9)
        # Predicted trajectory
        ax.plot(t, df_pred[pred_col], color=color, linewidth=1.5,
                linestyle="--", label="S-PINN (predicted)", alpha=0.85)

        ax.set_ylabel(title, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

        # Stress onset annotation
        ax.axvline(x=20.0, color="#95a5a6", linestyle="-.", linewidth=1.0,
                   alpha=0.7)
        if ax == axes[0]:
            ax.annotate(r"Stress onset $S(t)$", xy=(20.0, ax.get_ylim()[1] * 0.9),
                        fontsize=9, color="#7f8c8d", ha="left",
                        xytext=(22, ax.get_ylim()[1] * 0.85))

    axes[-1].set_xlabel("Time $t$", fontweight="bold")
    fig.suptitle("Neuro-Immuno-Endocrine Trajectory Cascade\n"
                 "SODE Ground Truth vs. S-PINN Predictions",
                 fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_phase_portrait(df_true, df_pred, output_path):
    """
    2D phase space plot of H(t) vs I(t) showing convergence
    to a stable homeostatic attractor.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    H_true = df_true["H"].values
    I_true = df_true["I"].values
    H_pred = df_pred["H_pred"].values
    I_pred = df_pred["I_pred"].values

    # True trajectory
    scatter_true = ax.scatter(H_true[::5], I_true[::5],
                              c=df_true["time"].values[::5],
                              cmap="viridis", s=8, alpha=0.6,
                              label="SODE trajectory", zorder=2)

    # Predicted trajectory
    ax.plot(H_pred, I_pred, color=COLORS["pred"], linewidth=1.5,
            linestyle="--", alpha=0.7, label="S-PINN trajectory", zorder=3)

    # Mark initial and final states
    ax.plot(H_true[0], I_true[0], "o", color=COLORS["N"], markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, label="Initial state",
            zorder=5)
    ax.plot(H_true[-1], I_true[-1], "*", color=COLORS["I"], markersize=14,
            markeredgecolor="black", markeredgewidth=1.0, label="Attractor",
            zorder=5)

    # Colorbar for time
    cbar = plt.colorbar(scatter_true, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Time $t$", fontsize=11)

    ax.set_xlabel(r"$H(t)$ -- Hormonal Concentration", fontweight="bold")
    ax.set_ylabel(r"$I(t)$ -- Immune Effector Activity", fontweight="bold")
    ax.set_title("Phase Portrait: Homeostatic Attractor Convergence\n"
                 r"$H(t)$ vs $I(t)$ Phase Space",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_loss_convergence(loss_history, output_path):
    """
    Logarithmic plot of data loss vs physics residual loss
    over training epochs.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    epochs = loss_history["epoch"]
    l_data = loss_history["loss_data"]
    l_phys = loss_history["loss_physics"]
    l_total = loss_history["loss_total"]

    ax.semilogy(epochs, l_data, color=COLORS["data"], linewidth=1.5,
                label=r"$\mathcal{L}_{\mathrm{data}}$ (MSE)", alpha=0.85)
    ax.semilogy(epochs, l_phys, color=COLORS["phys"], linewidth=1.5,
                label=r"$\mathcal{L}_{\mathrm{physics}}$ (Residual)", alpha=0.85)
    ax.semilogy(epochs, l_total, color=COLORS["true"], linewidth=2.0,
                linestyle="--", label=r"$\mathcal{L}_{\mathrm{total}}$", alpha=0.7)

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss (log scale)", fontweight="bold")
    ax.set_title("S-PINN Training Convergence\n"
                 r"$\mathcal{L}_{\mathrm{total}} = "
                 r"\mathcal{L}_{\mathrm{data}} + "
                 r"\lambda \cdot \mathcal{L}_{\mathrm{physics}}$",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def run(project_root: str = None):
    """Main entry point for plot generation."""
    if project_root is None:
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    traj_path   = os.path.join(project_root, "data", "sode_trajectories.csv")
    pred_path   = os.path.join(project_root, "data", "spinn_predictions.csv")
    loss_path   = os.path.join(project_root, "data", "loss_history.npy")
    fig_dir     = os.path.join(project_root, "outputs", "figures")

    print("[Phase 4b] Generating publication figures...")

    # Load data
    df_true = pd.read_csv(traj_path)
    df_pred = pd.read_csv(pred_path)
    loss_history = np.load(loss_path, allow_pickle=True).item()

    # Generate figures
    plot_trajectory_cascade(
        df_true, df_pred,
        os.path.join(fig_dir, "trajectory_cascade.png")
    )

    plot_phase_portrait(
        df_true, df_pred,
        os.path.join(fig_dir, "phase_portrait.png")
    )

    plot_loss_convergence(
        loss_history,
        os.path.join(fig_dir, "loss_convergence.png")
    )

    print("[Phase 4b] Complete.\n")


if __name__ == "__main__":
    run()
