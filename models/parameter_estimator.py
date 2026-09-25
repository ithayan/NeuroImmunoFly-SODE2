"""
Parameter Estimation for the Extended SODE System
===================================================
Fits SODE parameters to experimental time-series data using both
gradient-based (torch) and derivative-free (scipy) optimization.

Supports:
  - Inverse problem: given observed trajectories, estimate ODE parameters
  - Bounded optimization: biological parameter constraints
  - Bootstrap confidence intervals
  - Synthetic experimental data generation for validation

The estimator minimizes a composite objective:
  L = L_trajectory(theta) + lambda_reg * L_regularization(theta)

where L_trajectory is the weighted MSE between simulated and observed
trajectories, and L_regularization penalizes deviation from prior
parameter estimates.

Output:
  evaluation/fitted_params.json  -- estimated parameters + confidence intervals
"""

import os
import json
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Optional, Tuple, List

from models.sode_extended import (
    PARAMS_V2, STATE_NAMES, T_START, T_END, DT, T_STRESS, X0_V2,
    euler_maruyama_extended, solve_deterministic_extended
)


# Parameter bounds (biologically plausible ranges)
PARAM_BOUNDS = {
    "alpha":    (0.1,  5.0),
    "gamma":    (0.01, 2.0),
    "kappa_f":  (0.01, 1.0),
    "k1":       (0.05, 2.0),
    "delta1":   (0.01, 1.0),
    "eta":      (0.01, 0.5),
    "k2":       (0.05, 2.0),
    "beta_I":   (0.0,  1.0),
    "delta2":   (0.01, 1.0),
    "mu":       (0.01, 0.5),
    "k3":       (0.05, 1.5),
    "lambda_c": (0.0,  2.0),
    "delta3":   (0.05, 1.0),
    "k4":       (0.05, 1.5),
    "delta4":   (0.01, 1.0),
    "psi":      (0.01, 0.3),
}

# Which parameters to fit (others are held fixed)
FIT_PARAMS = [
    "alpha", "gamma", "kappa_f", "k1", "delta1", "eta",
    "k2", "beta_I", "delta2", "mu", "k3", "delta3", "k4", "delta4",
]

# Per-compartment weights for trajectory fitting
COMPARTMENT_WEIGHTS = {
    "N": 1.0,
    "H": 1.0,
    "I": 1.5,   # prioritize immune dynamics
    "C": 0.8,
    "F": 0.8,
}


def generate_synthetic_experimental_data(
    params: dict,
    t_span: np.ndarray,
    X0: list,
    noise_level: float = 0.05,
    sampling_interval: int = 10,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 'experimental' data by solving the SODE with known
    parameters and adding measurement noise.

    Simulates realistic experimental conditions:
    - Sparse temporal sampling (not every timestep)
    - Gaussian measurement noise
    - Only observable compartments (N, H, I, C, F -- not G1, G2)

    Returns
    -------
    t_obs : ndarray, shape (n_obs,)
        Observation time points
    X_obs : ndarray, shape (n_obs, 5)
        Observed [N, H, I, C, F] with noise
    """
    rng = np.random.default_rng(seed)

    X_true = solve_deterministic_extended(X0, t_span, params)

    # Subsample (sparse observations)
    obs_idx = np.arange(0, len(t_span), sampling_interval)
    t_obs = t_span[obs_idx]
    X_obs_clean = X_true[obs_idx, :5]  # only observable compartments

    # Add measurement noise (proportional to signal magnitude)
    noise = rng.normal(0, noise_level, size=X_obs_clean.shape)
    X_obs = X_obs_clean * (1.0 + noise)
    X_obs = np.maximum(X_obs, 0.0)

    return t_obs, X_obs


def _params_from_vector(theta: np.ndarray, base_params: dict,
                        fit_names: list) -> dict:
    """Reconstruct full parameter dict from optimization vector."""
    params = base_params.copy()
    for i, name in enumerate(fit_names):
        params[name] = float(theta[i])
    return params


def _vector_from_params(params: dict, fit_names: list) -> np.ndarray:
    return np.array([params[name] for name in fit_names])


def trajectory_objective(
    theta: np.ndarray,
    t_obs: np.ndarray,
    X_obs: np.ndarray,
    t_full: np.ndarray,
    obs_indices: np.ndarray,
    base_params: dict,
    fit_names: list,
    X0: list,
    lambda_reg: float = 0.01,
    prior_theta: Optional[np.ndarray] = None,
) -> float:
    """
    Objective function: weighted MSE between simulated and observed
    trajectories + regularization toward prior parameter estimates.
    """
    params = _params_from_vector(theta, base_params, fit_names)

    try:
        X_sim = solve_deterministic_extended(X0, t_full, params)
    except Exception:
        return 1e10

    X_sim_obs = X_sim[obs_indices, :5]

    if np.any(np.isnan(X_sim_obs)) or np.any(np.isinf(X_sim_obs)):
        return 1e10

    # Weighted MSE per compartment
    loss = 0.0
    comp_names = ["N", "H", "I", "C", "F"]
    for j, name in enumerate(comp_names):
        w = COMPARTMENT_WEIGHTS.get(name, 1.0)
        scale = max(np.std(X_obs[:, j]), 1e-6)
        loss += w * np.mean(((X_sim_obs[:, j] - X_obs[:, j]) / scale) ** 2)

    # L2 regularization toward prior
    if prior_theta is not None and lambda_reg > 0:
        loss += lambda_reg * np.sum(((theta - prior_theta) / (prior_theta + 1e-6)) ** 2)

    return float(loss)


def fit_parameters_lbfgsb(
    t_obs: np.ndarray,
    X_obs: np.ndarray,
    base_params: Optional[dict] = None,
    fit_names: Optional[list] = None,
    X0: Optional[list] = None,
    lambda_reg: float = 0.01,
    maxiter: int = 500,
) -> Dict:
    """
    Fit SODE parameters using L-BFGS-B (bounded quasi-Newton).

    Parameters
    ----------
    t_obs : ndarray, shape (n_obs,)
    X_obs : ndarray, shape (n_obs, 5)
        Observed [N, H, I, C, F]
    base_params : dict
        Starting parameter values (also used for non-fitted params)
    fit_names : list
        Parameter names to optimize
    X0 : list
        Initial conditions
    lambda_reg : float
        Regularization strength

    Returns
    -------
    result : dict with keys 'params', 'loss', 'convergence'
    """
    if base_params is None:
        base_params = PARAMS_V2.copy()
    if fit_names is None:
        fit_names = FIT_PARAMS
    if X0 is None:
        X0 = X0_V2

    t_full = np.arange(T_START, T_END + DT, DT)
    obs_indices = np.array([
        np.argmin(np.abs(t_full - t)) for t in t_obs
    ])

    theta0 = _vector_from_params(base_params, fit_names)
    bounds = [PARAM_BOUNDS.get(name, (1e-4, 10.0)) for name in fit_names]

    result = minimize(
        trajectory_objective,
        theta0,
        args=(t_obs, X_obs, t_full, obs_indices, base_params,
              fit_names, X0, lambda_reg, theta0),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10},
    )

    fitted_params = _params_from_vector(result.x, base_params, fit_names)

    return {
        "params": fitted_params,
        "fit_names": fit_names,
        "fitted_values": {name: fitted_params[name] for name in fit_names},
        "loss": float(result.fun),
        "convergence": bool(result.success),
        "message": result.message,
        "n_iterations": int(result.nit),
    }


def fit_parameters_de(
    t_obs: np.ndarray,
    X_obs: np.ndarray,
    base_params: Optional[dict] = None,
    fit_names: Optional[list] = None,
    X0: Optional[list] = None,
    lambda_reg: float = 0.01,
    maxiter: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Global parameter estimation using Differential Evolution.
    More robust to local minima than L-BFGS-B.
    """
    if base_params is None:
        base_params = PARAMS_V2.copy()
    if fit_names is None:
        fit_names = FIT_PARAMS
    if X0 is None:
        X0 = X0_V2

    t_full = np.arange(T_START, T_END + DT, DT)
    obs_indices = np.array([
        np.argmin(np.abs(t_full - t)) for t in t_obs
    ])

    theta0 = _vector_from_params(base_params, fit_names)
    bounds = [PARAM_BOUNDS.get(name, (1e-4, 10.0)) for name in fit_names]

    result = differential_evolution(
        trajectory_objective,
        bounds=bounds,
        args=(t_obs, X_obs, t_full, obs_indices, base_params,
              fit_names, X0, lambda_reg, theta0),
        maxiter=maxiter,
        seed=seed,
        tol=1e-8,
        polish=True,
    )

    fitted_params = _params_from_vector(result.x, base_params, fit_names)

    return {
        "params": fitted_params,
        "fit_names": fit_names,
        "fitted_values": {name: fitted_params[name] for name in fit_names},
        "loss": float(result.fun),
        "convergence": bool(result.success),
        "message": result.message,
        "n_iterations": int(result.nit),
    }


def bootstrap_confidence_intervals(
    t_obs: np.ndarray,
    X_obs: np.ndarray,
    n_bootstrap: int = 50,
    base_params: Optional[dict] = None,
    fit_names: Optional[list] = None,
    X0: Optional[list] = None,
    seed: int = 42,
) -> Dict:
    """
    Nonparametric bootstrap for parameter confidence intervals.
    Resamples observation time points with replacement.
    """
    if base_params is None:
        base_params = PARAMS_V2.copy()
    if fit_names is None:
        fit_names = FIT_PARAMS
    if X0 is None:
        X0 = X0_V2

    rng = np.random.default_rng(seed)
    n_obs = len(t_obs)
    bootstrap_params = {name: [] for name in fit_names}

    print(f"  Running {n_bootstrap} bootstrap iterations...")

    for b in range(n_bootstrap):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        idx = np.sort(np.unique(idx))

        t_boot = t_obs[idx]
        X_boot = X_obs[idx]

        try:
            result = fit_parameters_lbfgsb(
                t_boot, X_boot, base_params, fit_names, X0,
                lambda_reg=0.02, maxiter=200,
            )
            if result["convergence"]:
                for name in fit_names:
                    bootstrap_params[name].append(result["fitted_values"][name])
        except Exception:
            continue

        if (b + 1) % 10 == 0:
            print(f"    Bootstrap {b + 1}/{n_bootstrap} complete")

    ci = {}
    for name in fit_names:
        vals = np.array(bootstrap_params[name])
        if len(vals) >= 5:
            ci[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_2.5": float(np.percentile(vals, 2.5)),
                "ci_97.5": float(np.percentile(vals, 97.5)),
                "n_successful": len(vals),
            }
        else:
            ci[name] = {"mean": float("nan"), "note": "insufficient samples"}

    return ci


def run(project_root: str = None):
    """
    Demonstration: generate synthetic experimental data from known
    parameters, then recover them via parameter estimation.
    """
    if project_root is None:
        project_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."
        )

    print("[Phase 2b] Parameter estimation (synthetic validation)...")

    t_span = np.arange(T_START, T_END + DT, DT)

    # Generate synthetic experimental data with known parameters
    print("  Generating synthetic experimental data...")
    t_obs, X_obs = generate_synthetic_experimental_data(
        PARAMS_V2, t_span, X0_V2,
        noise_level=0.05, sampling_interval=20, seed=123,
    )
    print(f"  Observation points: {len(t_obs)}")
    print(f"  Observable compartments: N, H, I, C, F")

    # Perturb parameters as starting point (simulate unknown parameters)
    perturbed = PARAMS_V2.copy()
    rng = np.random.default_rng(99)
    for name in FIT_PARAMS:
        perturbed[name] *= rng.uniform(0.7, 1.3)

    # Fit using L-BFGS-B
    print("\n  Fitting with L-BFGS-B...")
    result_lbfgs = fit_parameters_lbfgsb(
        t_obs, X_obs, base_params=perturbed, maxiter=500
    )
    print(f"  Loss: {result_lbfgs['loss']:.6f}  "
          f"Converged: {result_lbfgs['convergence']}")

    # Compare fitted vs true
    print("\n  Parameter recovery (true -> fitted):")
    for name in FIT_PARAMS:
        true_val = PARAMS_V2[name]
        fit_val = result_lbfgs["fitted_values"][name]
        pct_err = abs(fit_val - true_val) / (true_val + 1e-8) * 100
        print(f"    {name:12s}: {true_val:.4f} -> {fit_val:.4f}  "
              f"(error: {pct_err:.1f}%)")

    # Bootstrap confidence intervals (reduced iterations for demo)
    print("\n  Computing bootstrap confidence intervals...")
    ci = bootstrap_confidence_intervals(
        t_obs, X_obs, n_bootstrap=30,
        base_params=result_lbfgs["params"],
    )

    # Save results
    output = {
        "method": "L-BFGS-B",
        "fitted_params": result_lbfgs["fitted_values"],
        "true_params": {name: PARAMS_V2[name] for name in FIT_PARAMS},
        "loss": result_lbfgs["loss"],
        "converged": result_lbfgs["convergence"],
        "confidence_intervals": ci,
    }

    out_path = os.path.join(project_root, "evaluation", "fitted_params.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")
    print("[Phase 2b] Complete.\n")

    return output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    run()
