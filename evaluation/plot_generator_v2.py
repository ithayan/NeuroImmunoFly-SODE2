"""
Extended Figure Generator for NeuroImmunoFly-SODE v2
======================================================
Generates publication-grade figures for the 7-variable system.

Figures:
  1. trajectory_cascade_v2.png  -- 7-panel true vs predicted
  2. circadian_overlay.png      -- compartments overlaid with clock
  3. phase_portrait_v2.png      -- H-I phase space + bistability
  4. feedback_diagram.png       -- feedback loop correlation heatmap
  5. loss_convergence_v2.png    -- loss + adaptive lambda schedule

Output directory: outputs/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "true":   "#2c3e50",
    "pred":   "#e74c3c",
    "N":      "#3498db",
    "H":      "#e67e22",
    "I":      "#27ae60",
    "C":      "#9b59b6",
    "F":      "#e74c3c",
    "G1":     "#1abc9c",
    "G2":     "#34495e",
    "clock":  "#f39c12",
    "data_l": "#8e44ad",
    "phys_l": "#c0392b",
}

STATE_NAMES = ["N", "H", "I", "C", "F", "G1", "G2"]
STATE_LABELS = {
    "N":  r"$N(t)$ -- Neuronal",
    "H":  r"$H(t)$ -- Hormonal (DH44)",
    "I":  r"$I(t)$ -- Immune",
    "C":  r"$C(t)$ -- Cytokine",
    "F":  r"$F(t)$ -- Stress Hormone",
    "G1": r"$G_1(t)$ -- Clock (cos)",
    "G2": r"$G_2(t)$ -- Clock (sin)",
}


def plot_trajectory_cascade_v2(df_true, df_pred, output_path):
    fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=True)

    t = df_true["time"].values

    for ax, name in zip(axes, STATE_NAMES):
        color = COLORS.get(name, "#333333")

        ax.plot(t, df_true[name], color=COLORS["true"], linewidth=1.5,
                label="SODE (ground truth)", alpha=0.9)
        ax.plot(t, df_pred[f"{name}_pred"], color=color, linewidth=1.2,
                linestyle="--", label="S-PINN (predicted)", alpha=0.85)

        ax.set_ylabel(STATE_LABELS[name], fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.25, linestyle=":")

        ax.axvline(x=20.0, color="#95a5a6", linestyle="-.", linewidth=0.8,
                   alpha=0.6)

    axes[0].annotate(r"Stress onset $S(t)$", xy=(20.5, axes[0].get_ylim()[1] * 0.85),
                     fontsize=8, color="#7f8c8d")
    axes[-1].set_xlabel("Time $t$", fontweight="bold")

    fig.suptitle("Extended Neuro-Immuno-Endocrine Trajectory Cascade\n"
                 "7-Variable SODE vs. Extended S-PINN",
                 fontsize=13, fontweight="bold", y=1.005)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_circadian_overlay(df_true, output_path):
    """
    Overlay N, H, I dynamics with circadian clock output Phi(t).
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    t = df_true["time"].values

    # Circadian output
    if "Phi" in df_true.columns:
        Phi = df_true["Phi"].values
    else:
        from models.sode_extended import circadian_output, PARAMS_V2
        Phi = np.array([circadian_output(g1, PARAMS_V2["mu_c"])
                        for g1 in df_true["G1"].values])

    ax1.fill_between(t, 0, Phi, alpha=0.15, color=COLORS["clock"],
                     label=r"$\Phi(t)$ Circadian output")
    ax1.set_ylabel(r"$\Phi(t)$ (Circadian)", color=COLORS["clock"],
                   fontweight="bold")
    ax1.set_ylim(-0.05, 1.1)
    ax1.tick_params(axis="y", labelcolor=COLORS["clock"])

    ax2 = ax1.twinx()

    # Normalize N, H, I to [0,1] for overlay
    for name, color in [("N", COLORS["N"]), ("H", COLORS["H"]),
                        ("I", COLORS["I"])]:
        vals = df_true[name].values
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            normed = (vals - vmin) / (vmax - vmin)
        else:
            normed = vals
        ax2.plot(t, normed, color=color, linewidth=1.3, alpha=0.85,
                 label=f"{name}(t) (normalized)")

    ax2.set_ylabel("Normalized State", fontweight="bold")
    ax2.set_ylim(-0.05, 1.3)

    ax1.axvline(x=20.0, color="#95a5a6", linestyle="-.", linewidth=0.8)
    ax1.set_xlabel("Time $t$", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               framealpha=0.9)

    ax1.set_title("Circadian Modulation of Neuro-Immuno-Endocrine Dynamics",
                  fontweight="bold")
    ax1.grid(True, alpha=0.2, linestyle=":")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_phase_portrait_v2(df_true, df_pred, output_path):
    """
    H-I phase portrait with bistability annotation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    H_true = df_true["H"].values
    I_true = df_true["I"].values
    H_pred = df_pred["H_pred"].values
    I_pred = df_pred["I_pred"].values
    t = df_true["time"].values

    # Left: H-I phase space
    ax = axes[0]
    sc = ax.scatter(H_true[::3], I_true[::3], c=t[::3], cmap="viridis",
                    s=6, alpha=0.6, zorder=2)
    ax.plot(H_pred, I_pred, color=COLORS["pred"], linewidth=1.2,
            linestyle="--", alpha=0.7, label="S-PINN", zorder=3)
    ax.plot(H_true[0], I_true[0], "o", color=COLORS["N"], markersize=10,
            markeredgecolor="k", markeredgewidth=1.2, label="Start", zorder=5)
    ax.plot(H_true[-1], I_true[-1], "*", color=COLORS["I"], markersize=12,
            markeredgecolor="k", label="Attractor", zorder=5)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Time $t$")
    ax.set_xlabel(r"$H(t)$ -- Hormonal", fontweight="bold")
    ax.set_ylabel(r"$I(t)$ -- Immune", fontweight="bold")
    ax.set_title("H-I Phase Portrait", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")

    # Right: F-N phase space (negative feedback)
    ax = axes[1]
    F_true = df_true["F"].values
    N_true = df_true["N"].values
    sc2 = ax.scatter(F_true[::3], N_true[::3], c=t[::3], cmap="magma",
                     s=6, alpha=0.6, zorder=2)
    ax.plot(F_true[0], N_true[0], "o", color=COLORS["N"], markersize=10,
            markeredgecolor="k", markeredgewidth=1.2, label="Start", zorder=5)
    ax.plot(F_true[-1], N_true[-1], "*", color=COLORS["F"], markersize=12,
            markeredgecolor="k", label="Steady state", zorder=5)

    cbar2 = plt.colorbar(sc2, ax=ax, shrink=0.85)
    cbar2.set_label("Time $t$")
    ax.set_xlabel(r"$F(t)$ -- Stress Hormone", fontweight="bold")
    ax.set_ylabel(r"$N(t)$ -- Neuronal", fontweight="bold")
    ax.set_title("F-N Phase Portrait (Negative Feedback)", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle("Phase Space Analysis: Attractor Dynamics & Feedback",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_feedback_heatmap(df_true, output_path):
    """
    Cross-correlation heatmap showing feedback loop strengths.
    """
    comp_names = ["N", "H", "I", "C", "F"]
    n = len(comp_names)
    data = np.column_stack([df_true[c].values for c in comp_names])

    # Lagged correlation matrix (lag=10 timesteps)
    lag = 10
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = np.corrcoef(
                data[:-lag, i], data[lag:, j]
            )[0, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(comp_names, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(comp_names, fontweight="bold")
    ax.set_xlabel("Target (lagged)", fontweight="bold")
    ax.set_ylabel("Source", fontweight="bold")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Lagged Correlation", fontsize=11)

    ax.set_title("Feedback Loop Strength (Lagged Cross-Correlation)\n"
                 "Source(t) vs Target(t + lag)",
                 fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_loss_convergence_v2(loss_history, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    epochs = loss_history["epoch"]
    l_data = loss_history["loss_data"]
    l_phys = loss_history["loss_physics"]
    l_total = loss_history["loss_total"]
    lam = loss_history.get("lambda_phys", [1.0] * len(epochs))

    ax1.semilogy(epochs, l_data, color=COLORS["data_l"], linewidth=1.3,
                 label=r"$\mathcal{L}_{\mathrm{data}}$", alpha=0.85)
    ax1.semilogy(epochs, l_phys, color=COLORS["phys_l"], linewidth=1.3,
                 label=r"$\mathcal{L}_{\mathrm{physics}}$", alpha=0.85)
    ax1.semilogy(epochs, l_total, color=COLORS["true"], linewidth=1.8,
                 linestyle="--", label=r"$\mathcal{L}_{\mathrm{total}}$",
                 alpha=0.7)

    ax1.set_ylabel("Loss (log scale)", fontweight="bold")
    ax1.set_title("Extended S-PINN Training Convergence\n"
                  r"$\mathcal{L}_{\mathrm{total}} = "
                  r"\mathcal{L}_{\mathrm{data}} + "
                  r"\lambda(e) \cdot \mathcal{L}_{\mathrm{physics}}$",
                  fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle=":", which="both")

    ax2.plot(epochs, lam, color=COLORS["clock"], linewidth=1.5)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel(r"$\lambda(e)$", fontweight="bold", color=COLORS["clock"])
    ax2.set_title("Adaptive Physics Weight (Curriculum Schedule)",
                  fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle=":")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def run(project_root: str = None):
    if project_root is None:
        project_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."
        )

    traj_path = os.path.join(project_root, "data", "sode_trajectories_v2.csv")
    pred_path = os.path.join(project_root, "data", "spinn_predictions_v2.csv")
    loss_path = os.path.join(project_root, "data", "loss_history_v2.npy")
    fig_dir   = os.path.join(project_root, "outputs", "figures")

    print("[Phase 4b v2] Generating extended publication figures...")

    df_true = pd.read_csv(traj_path)
    df_pred = pd.read_csv(pred_path)
    loss_history = np.load(loss_path, allow_pickle=True).item()

    plot_trajectory_cascade_v2(
        df_true, df_pred,
        os.path.join(fig_dir, "trajectory_cascade_v2.png")
    )

    plot_circadian_overlay(
        df_true,
        os.path.join(fig_dir, "circadian_overlay.png")
    )

    plot_phase_portrait_v2(
        df_true, df_pred,
        os.path.join(fig_dir, "phase_portrait_v2.png")
    )

    plot_feedback_heatmap(
        df_true,
        os.path.join(fig_dir, "feedback_diagram.png")
    )

    plot_loss_convergence_v2(
        loss_history,
        os.path.join(fig_dir, "loss_convergence_v2.png")
    )

    print("[Phase 4b v2] Complete.\n")


if __name__ == "__main__":
    run()
