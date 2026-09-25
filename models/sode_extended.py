"""
Extended Stochastic ODE System with Multi-Timescale & Feedback Dynamics
========================================================================
Expands the 3-compartment SODE to a 7-variable system incorporating:
  - Circadian oscillator (Poincare limit cycle)
  - Cytokine-mediated feedback signaling
  - Stress hormone with negative feedback to neurons
  - Bistable immune switching (Hill-function positive feedback)
  - State-dependent stochastic noise

State vector X(t) = [N, H, I, C, F, G1, G2]:
  N(t)  : Neuronal activity (circadian-modulated, stress-suppressed)
  H(t)  : Hormonal / DH44-CRH homolog (cytokine-modulated)
  I(t)  : Immune effectors (bistable: acute vs. chronic)
  C(t)  : Cytokine signaling (immune-produced, circadian-gated)
  F(t)  : Stress hormone / cortisol analog (negative feedback to N)
  G1(t) : Circadian oscillator variable 1
  G2(t) : Circadian oscillator variable 2

Governing Equations:

  Circadian (Poincare limit cycle, period T_circ):
    dG1/dt = mu_c * G1 - omega * G2 - G1 * (G1^2 + G2^2)
    dG2/dt = omega * G1 + mu_c * G2 - G2 * (G1^2 + G2^2)
    Phi(t) = (1 + G1(t) / sqrt(mu_c)) / 2   (clock output in [0,1])

  Neuronal:
    dN/dt = alpha * Phi(t) - gamma * N * S(t) - kappa_f * F * N / (K_N + N)
            + sigma_N * dW_N

  Hormonal:
    dH/dt = k1 * N * Phi(t) - delta1 * H - eta * C * H / (K_H + H)

  Immune (bistable):
    dI/dt = k2 * H + beta_I * I^2 / (K_b^2 + I^2) - delta2 * I - mu * I^2
            + sigma_I * dW_I

  Cytokine:
    dC/dt = k3 * I * (1 + lambda_c * (1 - Phi(t))) - delta3 * C

  Stress hormone:
    dF/dt = k4 * H - delta4 * F - psi * F^2

Output:
  data/sode_trajectories_v2.csv  -- [t, N, H, I, C, F, G1, G2, Phi]
"""

import os
import numpy as np
import pandas as pd

# =====================================================================
# Extended SODE Parameters
# =====================================================================
PARAMS_V2 = {
    # --- Neuronal ---
    "alpha":     1.0,     # basal neuronal rate
    "gamma":     0.3,     # stress-coupled suppression
    "kappa_f":   0.15,    # stress hormone negative feedback strength
    "K_N":       0.5,     # half-saturation for N feedback
    "sigma_N":   0.02,    # neuronal noise intensity

    # --- Hormonal ---
    "k1":        0.5,     # N -> H coupling
    "delta1":    0.1,     # H degradation
    "eta":       0.08,    # cytokine modulation of H
    "K_H":       0.3,     # half-saturation for H modulation

    # --- Immune (bistable) ---
    "k2":        0.4,     # H -> I activation
    "beta_I":    0.25,    # Hill positive feedback amplitude
    "K_b":       1.2,     # Hill half-activation threshold
    "delta2":    0.15,    # I linear clearance
    "mu":        0.05,    # quadratic homeostatic dampening
    "sigma_I":   0.015,   # immune noise intensity

    # --- Cytokine ---
    "k3":        0.3,     # I -> C production
    "lambda_c":  0.6,     # circadian gating amplitude for cytokines
    "delta3":    0.2,     # C clearance

    # --- Stress hormone ---
    "k4":        0.35,    # H -> F production
    "delta4":    0.12,    # F linear clearance
    "psi":       0.04,    # F quadratic clearance

    # --- Circadian oscillator ---
    "mu_c":      1.0,     # limit cycle amplitude parameter
    "T_circ":    24.0,    # circadian period (hours)
}

# Temporal parameters
T_START  = 0.0
T_END    = 120.0   # extended to capture circadian effects
DT       = 0.05    # finer timestep for oscillator stability
T_STRESS = 20.0


# Initial conditions [N, H, I, C, F, G1, G2]
X0_V2 = [1.0, 0.5, 0.2, 0.1, 0.05, 1.0, 0.0]

STATE_NAMES = ["N", "H", "I", "C", "F", "G1", "G2"]


def stress_function(t: float, t_onset: float = T_STRESS) -> float:
    return 1.0 if t >= t_onset else 0.0


def circadian_output(G1: float, mu_c: float) -> float:
    """Clock output Phi(t) in [0, 1]."""
    amplitude = np.sqrt(max(mu_c, 1e-8))
    return np.clip((1.0 + G1 / amplitude) / 2.0, 0.0, 1.0)


def sode_extended_drift(X, t, params):
    """
    Deterministic drift for the 7-variable extended system.

    Parameters
    ----------
    X : array_like, shape (7,)
        [N, H, I, C, F, G1, G2]
    t : float
    params : dict

    Returns
    -------
    dXdt : list of 7 floats
    """
    N, H, I, C, F, G1, G2 = X
    S = stress_function(t)
    mu_c = params["mu_c"]
    omega = 2.0 * np.pi / params["T_circ"]
    Phi = circadian_output(G1, mu_c)

    # Circadian oscillator (Poincare)
    r_sq = G1**2 + G2**2
    dG1 = mu_c * G1 - omega * G2 - G1 * r_sq
    dG2 = omega * G1 + mu_c * G2 - G2 * r_sq

    # Neuronal (circadian-modulated, stress-suppressed, hormone-feedback)
    dN = (params["alpha"] * Phi
          - params["gamma"] * N * S
          - params["kappa_f"] * F * N / (params["K_N"] + N + 1e-8))

    # Hormonal (cytokine-modulated)
    dH = (params["k1"] * N * Phi
          - params["delta1"] * H
          - params["eta"] * C * H / (params["K_H"] + H + 1e-8))

    # Immune (bistable Hill + quadratic dampening)
    hill_term = params["beta_I"] * I**2 / (params["K_b"]**2 + I**2 + 1e-8)
    dI = (params["k2"] * H
          + hill_term
          - params["delta2"] * I
          - params["mu"] * I**2)

    # Cytokine (circadian-gated production)
    dC = (params["k3"] * I * (1.0 + params["lambda_c"] * (1.0 - Phi))
          - params["delta3"] * C)

    # Stress hormone (negative feedback source)
    dF = (params["k4"] * H
          - params["delta4"] * F
          - params["psi"] * F**2)

    return [dN, dH, dI, dC, dF, dG1, dG2]


def euler_maruyama_extended(X0, t_span, params, seed=42):
    """
    Euler-Maruyama solver for the extended 7-variable SODE.
    State-dependent noise on N and I compartments.
    """
    rng = np.random.default_rng(seed)
    n_steps = len(t_span)
    dt = t_span[1] - t_span[0]
    n_vars = 7

    X = np.zeros((n_steps, n_vars))
    X[0] = X0

    for i in range(1, n_steps):
        t = t_span[i - 1]
        state = X[i - 1]
        drift = sode_extended_drift(state, t, params)

        # Wiener increments for stochastic compartments
        dW_N = rng.normal(0, np.sqrt(dt))
        dW_I = rng.normal(0, np.sqrt(dt))

        # Euler-Maruyama update
        for j in range(n_vars):
            X[i, j] = state[j] + drift[j] * dt

        # Add noise to N and I
        X[i, 0] += params["sigma_N"] * dW_N
        X[i, 2] += params["sigma_I"] * dW_I

        # Enforce non-negativity (biological constraint)
        # G1, G2 can be negative (oscillator), others cannot
        X[i, :5] = np.maximum(X[i, :5], 0.0)

    return X


def solve_deterministic_extended(X0, t_span, params):
    """
    Deterministic solution (sigma=0) using scipy odeint.
    """
    from scipy.integrate import odeint

    def _drift_wrapper(X, t):
        return sode_extended_drift(X, t, params)

    return odeint(_drift_wrapper, X0, t_span)


def run(output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        )

    t_span = np.arange(T_START, T_END + DT, DT)

    print("[Phase 2 v2] Running extended SODE simulation (7 variables)...")
    print(f"  Time domain  : [{T_START}, {T_END}], dt={DT}")
    print(f"  Stress onset : t = {T_STRESS}")
    print(f"  Circadian T  : {PARAMS_V2['T_circ']} h")
    print(f"  Variables    : {STATE_NAMES}")

    X_stoch = euler_maruyama_extended(X0_V2, t_span, PARAMS_V2, seed=42)
    X_det = solve_deterministic_extended(X0_V2, t_span, PARAMS_V2)

    # Compute circadian output
    mu_c = PARAMS_V2["mu_c"]
    Phi_det = np.array([circadian_output(g1, mu_c) for g1 in X_det[:, 5]])
    Phi_stoch = np.array([circadian_output(g1, mu_c) for g1 in X_stoch[:, 5]])

    df = pd.DataFrame({"time": t_span})
    for j, name in enumerate(STATE_NAMES):
        df[name] = X_det[:, j]
        df[f"{name}_stoch"] = X_stoch[:, j]
    df["Phi"] = Phi_det
    df["Phi_stoch"] = Phi_stoch

    csv_path = os.path.join(output_dir, "sode_trajectories_v2.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Trajectories saved -> {csv_path}  shape={df.shape}")

    for name in STATE_NAMES:
        lo, hi = df[name].min(), df[name].max()
        print(f"  {name}(t) range: [{lo:.4f}, {hi:.4f}]")

    print("[Phase 2 v2] Complete.\n")
    return df


if __name__ == "__main__":
    run()
