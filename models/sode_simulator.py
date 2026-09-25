"""
Stochastic ODE Simulator for the Neuro-Immuno-Endocrine Stress Axis
====================================================================
Implements a multi-compartment SODE system modeling the coupled dynamics
of neuronal (N), hormonal (H), and immune (I) compartments under
external stress perturbation.

Governing Equations:
    dN/dt = alpha - gamma * N(t) * S(t) + xi(t)
    dH/dt = k_1 * N(t) - delta_1 * H(t)
    dI/dt = k_2 * H(t) - delta_2 * I(t) - mu * I(t)^2

where:
    S(t) = step function (stress onset at t_stress)
    xi(t) = Wiener process noise (stochastic driving term)

Output:
    data/sode_trajectories.csv  --  time-series of [t, N, H, I]
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# ---------- SODE Parameters ----------
# Biophysical rate constants (dimensionless, normalized)
PARAMS = {
    "alpha":    1.0,     # basal neuronal production rate
    "gamma":    0.3,     # stress-coupled neuronal suppression
    "k_1":      0.5,     # neuro -> hormonal coupling
    "delta_1":  0.1,     # hormonal degradation rate
    "k_2":      0.4,     # hormonal -> immune activation
    "delta_2":  0.15,    # immune clearance rate
    "mu":       0.05,    # quadratic homeostatic immune dampening
    "sigma":    0.02,    # noise intensity for xi(t)
}

# Temporal parameters
T_START   = 0.0
T_END     = 100.0
DT        = 0.1
T_STRESS  = 20.0   # stress onset time

# Initial conditions [N(0), H(0), I(0)]
X0 = [1.0, 0.5, 0.2]


def stress_function(t: float, t_onset: float = T_STRESS) -> float:
    """
    External stress signal S(t): Heaviside step function.
    Returns 1.0 for t >= t_onset, else 0.0.
    """
    return 1.0 if t >= t_onset else 0.0


def sode_deterministic(X, t, params):
    """
    Deterministic component of the SODE system (drift term).

    Parameters
    ----------
    X : array_like
        State vector [N, H, I]
    t : float
        Current time
    params : dict
        System parameters

    Returns
    -------
    dXdt : list
        Time derivatives [dN/dt, dH/dt, dI/dt]
    """
    N, H, I = X
    S = stress_function(t)

    dN_dt = params["alpha"] - params["gamma"] * N * S
    dH_dt = params["k_1"] * N - params["delta_1"] * H
    dI_dt = params["k_2"] * H - params["delta_2"] * I - params["mu"] * I**2

    return [dN_dt, dH_dt, dI_dt]


def euler_maruyama_solve(X0, t_span, params, seed=42):
    """
    Solve the SODE system using the Euler-Maruyama scheme for
    proper handling of stochastic terms.

    The Wiener increment: dW ~ N(0, dt)
    Noise term: xi(t) * dW = sigma * dW (applied to N compartment)

    Parameters
    ----------
    X0 : list
        Initial conditions [N0, H0, I0]
    t_span : ndarray
        Time array
    params : dict
        System parameters
    seed : int
        Random seed for reproducibility

    Returns
    -------
    trajectories : ndarray
        Shape (len(t_span), 3) -- columns are [N, H, I]
    """
    rng = np.random.default_rng(seed)
    n_steps = len(t_span)
    dt = t_span[1] - t_span[0]

    X = np.zeros((n_steps, 3))
    X[0] = X0

    sigma = params["sigma"]

    for i in range(1, n_steps):
        t = t_span[i - 1]
        N, H, I = X[i - 1]
        S = stress_function(t)

        # Deterministic drift
        dN = params["alpha"] - params["gamma"] * N * S
        dH = params["k_1"] * N - params["delta_1"] * H
        dI = params["k_2"] * H - params["delta_2"] * I - params["mu"] * I**2

        # Stochastic diffusion (Wiener increment)
        dW = rng.normal(0, np.sqrt(dt))

        # Euler-Maruyama update
        X[i, 0] = N + dN * dt + sigma * dW
        X[i, 1] = H + dH * dt
        X[i, 2] = I + dI * dt

        # Enforce non-negativity (biological constraint)
        X[i] = np.maximum(X[i], 0.0)

    return X


def solve_deterministic(X0, t_span, params):
    """
    Solve the deterministic ODE system using scipy.integrate.odeint.
    Used as a reference trajectory for PINN training.
    """
    solution = odeint(sode_deterministic, X0, t_span, args=(params,))
    return solution


def run(output_dir: str = None):
    """Main entry point for SODE simulation."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "data")

    t_span = np.arange(T_START, T_END + DT, DT)

    print("[Phase 2] Running SODE simulation...")
    print(f"  Time domain  : [{T_START}, {T_END}], dt={DT}")
    print(f"  Stress onset : t = {T_STRESS}")
    print(f"  Parameters   : {PARAMS}")

    # --- Stochastic solution (Euler-Maruyama) ---
    X_stoch = euler_maruyama_solve(X0, t_span, PARAMS, seed=42)

    # --- Deterministic reference (odeint) ---
    X_det = solve_deterministic(X0, t_span, PARAMS)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "time":   t_span,
        "N":      X_det[:, 0],
        "H":      X_det[:, 1],
        "I":      X_det[:, 2],
        "N_stoch": X_stoch[:, 0],
        "H_stoch": X_stoch[:, 1],
        "I_stoch": X_stoch[:, 2],
    })

    csv_path = os.path.join(output_dir, "sode_trajectories.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Trajectories saved -> {csv_path}  shape={df.shape}")

    # --- Summary statistics ---
    print(f"  N(t) range: [{df['N'].min():.4f}, {df['N'].max():.4f}]")
    print(f"  H(t) range: [{df['H'].min():.4f}, {df['H'].max():.4f}]")
    print(f"  I(t) range: [{df['I'].min():.4f}, {df['I'].max():.4f}]")
    print("[Phase 2] Complete.\n")

    return df


if __name__ == "__main__":
    run()
