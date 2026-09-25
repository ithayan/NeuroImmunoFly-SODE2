#!/usr/bin/env python3
"""
NeuroImmunoFly-SODE: Master Pipeline Orchestrator
===================================================
End-to-end execution of the neuro-immuno-endocrine stress axis
simulation pipeline combining Drosophila connectomics with
Stochastic Physics-Informed Neural Networks (S-PINN).

Supports two pipeline versions:
  v1 (original): 3-compartment [N, H, I] with synthetic connectome
  v2 (extended): 7-compartment [N, H, I, C, F, G1, G2] with
                 empirical connectome, circadian oscillator,
                 cytokine feedback, bistable immune switching,
                 stress hormone negative feedback, and parameter fitting

Usage:
    python main.py           # Run v1 (original) pipeline
    python main.py --v2      # Run v2 (extended) pipeline
    python main.py --all     # Run both pipelines
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_v1():
    """Execute the original 3-compartment pipeline."""
    print("=" * 70)
    print("  NeuroImmunoFly-SODE Pipeline  [v1 - Original]")
    print("  3-Compartment Neuro-Immuno-Endocrine Stress Axis")
    print("=" * 70)
    print()

    t_start = time.time()
    data_dir = os.path.join(PROJECT_ROOT, "data")

    # Phase 1: Connectome
    print("-" * 70)
    from data.generate_connectome import run as run_connectome
    G, A, X = run_connectome(output_dir=data_dir)
    assert os.path.exists(os.path.join(data_dir, "connectome_adj.npy"))

    # Phase 2: SODE
    print("-" * 70)
    from models.sode_simulator import run as run_sode
    df_sode = run_sode(output_dir=data_dir)
    assert os.path.exists(os.path.join(data_dir, "sode_trajectories.csv"))

    # Phase 3: S-PINN
    print("-" * 70)
    from models.spinn_engine import run as run_spinn
    model, loss_history = run_spinn(project_root=PROJECT_ROOT)
    assert os.path.exists(os.path.join(PROJECT_ROOT, "outputs", "weights", "spinn_model.pth"))

    # Phase 4a: Metrics
    print("-" * 70)
    from evaluation.compute_metrics import run as run_metrics
    metrics = run_metrics(project_root=PROJECT_ROOT)

    # Phase 4b: Figures
    print("-" * 70)
    from evaluation.plot_generator import run as run_plots
    run_plots(project_root=PROJECT_ROOT)

    elapsed = time.time() - t_start
    print("=" * 70)
    print(f"  v1 PIPELINE COMPLETE  ({elapsed:.1f}s / {elapsed/60:.1f} min)")
    print("=" * 70)
    print()


def run_v2():
    """Execute the extended 7-compartment pipeline with all improvements."""
    print("=" * 70)
    print("  NeuroImmunoFly-SODE Pipeline  [v2 - Extended]")
    print("  7-Variable System: Circadian + Feedback + Bistability")
    print("  Improvements:")
    print("    1. Empirical synaptic connectivity (malecns T-bar weights)")
    print("    2. Parameter estimation from experimental time-series")
    print("    3. Multi-timescale SODE with circadian & feedback loops")
    print("=" * 70)
    print()

    t_start = time.time()
    data_dir = os.path.join(PROJECT_ROOT, "data")

    # Phase 1 v2: Empirical Connectome
    print("-" * 70)
    from data.neuprint_connectome import run as run_connectome_v2
    G, A, X = run_connectome_v2(output_dir=data_dir)
    assert os.path.exists(os.path.join(data_dir, "connectome_adj_v2.npy"))

    # Phase 2 v2: Extended SODE (7 variables)
    print("-" * 70)
    from models.sode_extended import run as run_sode_v2
    df_sode = run_sode_v2(output_dir=data_dir)
    assert os.path.exists(os.path.join(data_dir, "sode_trajectories_v2.csv"))

    # Phase 2b: Parameter Estimation (synthetic validation)
    print("-" * 70)
    from models.parameter_estimator import run as run_param_est
    param_results = run_param_est(project_root=PROJECT_ROOT)

    # Phase 3 v2: Extended S-PINN Training
    print("-" * 70)
    from models.spinn_extended import run as run_spinn_v2
    model, loss_history = run_spinn_v2(project_root=PROJECT_ROOT)
    assert os.path.exists(os.path.join(
        PROJECT_ROOT, "outputs", "weights", "spinn_model_v2.pth"))

    # Phase 4a v2: Extended Metrics
    print("-" * 70)
    from evaluation.compute_metrics_v2 import run as run_metrics_v2
    metrics = run_metrics_v2(project_root=PROJECT_ROOT)

    # Phase 4b v2: Extended Figures
    print("-" * 70)
    from evaluation.plot_generator_v2 import run as run_plots_v2
    run_plots_v2(project_root=PROJECT_ROOT)

    elapsed = time.time() - t_start
    print("=" * 70)
    print(f"  v2 PIPELINE COMPLETE  ({elapsed:.1f}s / {elapsed/60:.1f} min)")
    print()
    print("  v2 Outputs:")
    print(f"    data/connectome_adj_v2.npy        : Empirical adjacency ({A.shape})")
    print(f"    data/sode_trajectories_v2.csv     : 7-var SODE trajectories ({df_sode.shape})")
    print(f"    data/spinn_predictions_v2.csv     : Extended S-PINN predictions")
    print(f"    evaluation/fitted_params.json     : Parameter estimation results")
    print(f"    evaluation/metrics_report_v2.json : Extended metrics")
    print(f"    outputs/figures/                  : 5 publication figures")
    print("=" * 70)

    return 0


def main():
    args = sys.argv[1:]

    if "--v2" in args:
        return run_v2()
    elif "--all" in args:
        run_v1()
        print("\n" + "=" * 70)
        print("  Switching to v2 pipeline...")
        print("=" * 70 + "\n")
        return run_v2()
    else:
        run_v1()
        return 0


if __name__ == "__main__":
    sys.exit(main())
