# NeuroImmunoFly-SODE

## Stochastic Physics-Informed Neural Network for Drosophila Neuro-Immuno-Endocrine Stress Axis Modeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

This project presents a computational framework integrating **Drosophila melanogaster connectomics** with **Stochastic Physics-Informed Neural Networks (S-PINNs)** to model the neuro-immuno-endocrine stress axis. The architecture maps polyadic chemical synapses from sensory pathways through the mushroom body to neurosecretory cells in the **pars intercerebralis** (PI) — the Drosophila homolog of the mammalian hypothalamus — and downstream systemic immune effectors.

The PI neurons, which produce **Drosophila insulin-like peptides (DILPs)** and **diuretic hormone 44 (DH44)** — a functional homolog of mammalian **corticotropin-releasing hormone (CRH)** — serve as the central integrative hub linking neuronal stress perception to hormonal and immune modulation. This architecture is analogous to the mammalian **Hypothalamic-Pituitary-Adrenal (HPA) axis**.

The simulation pipeline:
1. Generates a biologically-motivated directed connectivity graph (370 neurons, 4 tiers)
2. Solves coupled **Stochastic Ordinary Differential Equations (SODEs)** via Euler-Maruyama integration
3. Trains a **Physics-Informed Neural Network** constrained by the governing SODE dynamics through autograd-computed residuals
4. Evaluates model fidelity through RMSE, R², and physics violation metrics

---

## Connectome Architecture

The directed graph models four functional tiers of the Drosophila stress axis:

| Tier | Biological Analog | Nodes | Function |
|------|-------------------|-------|----------|
| T1 | Sensory neurons (antennal, gustatory, mechanosensory) | 50 | Environmental stress detection |
| T2 | Mushroom body (Kenyon cells, MB output neurons) | 200 | Associative integration & memory |
| T3 | Pars intercerebralis neurosecretory cells | 20 | CRH/DH44-homolog release, neuroendocrine coupling |
| T4 | Systemic immune effectors (hemocytes, fat body, AMP signaling) | 100 | Innate immune response modulation |

**Total: 370 nodes** with stochastic inter-tier and intra-tier connectivity based on empirically-inspired synapse density parameters.

Polyadic synapse connectivity probabilities:
- Sensory → Mushroom Body: *p* = 0.08
- Mushroom Body → PI neurons: *p* = 0.12
- PI neurons → Immune effectors: *p* = 0.15
- Feedback (Immune → PI): *p* = 0.05
- Intra-tier lateral: *p* = 0.03

---

## SODE Governing Equations

The system state vector **X**(t) = [*N*(t), *H*(t), *I*(t)] evolves according to the coupled stochastic differential equations:

### Neuronal Compartment (N)
$$
\frac{dN}{dt} = \alpha - \gamma \cdot N(t) \cdot S(t) + \xi(t)
$$

where *S*(t) is a Heaviside step function representing stress onset at *t* = 20:

$$
S(t) = \begin{cases} 0 & t < t_{\text{stress}} \\ 1 & t \geq t_{\text{stress}} \end{cases}
$$

and ξ(t) is a Wiener process noise term: ξ(t) ~ σ · dW(t)/dt

### Hormonal Compartment (H)
$$
\frac{dH}{dt} = k_1 \cdot N(t) - \delta_1 \cdot H(t)
$$

### Immune Compartment (I)
$$
\frac{dI}{dt} = k_2 \cdot H(t) - \delta_2 \cdot I(t) - \mu \cdot I(t)^2
$$

The quadratic term −μ·I² implements **homeostatic immune dampening**, preventing unbounded immune activation and enforcing return to a stable attractor in the H-I phase space.

### Parameter Table

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Basal neuronal rate | α | 1.0 | Constitutive neuronal activity |
| Stress coupling | γ | 0.3 | Stress-mediated neuronal suppression |
| Neuro-hormonal coupling | k₁ | 0.5 | N → H activation rate |
| Hormonal degradation | δ₁ | 0.1 | H clearance rate |
| Hormonal-immune coupling | k₂ | 0.4 | H → I activation rate |
| Immune clearance | δ₂ | 0.15 | I linear clearance |
| Homeostatic dampening | μ | 0.05 | Quadratic immune self-regulation |
| Noise intensity | σ | 0.02 | Wiener process amplitude |

---

## S-PINN Architecture

### Network Design

```
Input (1) → [Linear(1,64) → Tanh] × 4 → Linear(64,3) → Output (3)
         t                                            [N̂, Ĥ, Î]
```

- **Input**: Normalized time *t* ∈ [0, 1]
- **Hidden layers**: 4 fully-connected layers with 64 neurons each, Tanh activation
- **Output**: Predicted compartment states [N̂(t), Ĥ(t), Î(t)]
- **Initialization**: Xavier normal

### Physics-Informed Loss Function

The total loss is a composite of data fidelity and physics constraint terms:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}
$$

**Data Loss** (supervised MSE):
$$
\mathcal{L}_{\text{data}} = \frac{1}{N_d} \sum_{i=1}^{N_d} \left[ (\hat{N}_i - N_i)^2 + (\hat{H}_i - H_i)^2 + (\hat{I}_i - I_i)^2 \right]
$$

**Physics Residual Loss** (autograd-enforced ODE compliance):
$$
\mathcal{L}_{\text{physics}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \left[ r_N^2(t_j) + r_H^2(t_j) + r_I^2(t_j) \right]
$$

where the residuals are computed using `torch.autograd.grad`:

$$
r_N(t) = \frac{\partial \hat{N}}{\partial t} - \left(\alpha - \gamma \hat{N} \cdot S(t)\right)
$$

$$
r_H(t) = \frac{\partial \hat{H}}{\partial t} - \left(k_1 \hat{N} - \delta_1 \hat{H}\right)
$$

$$
r_I(t) = \frac{\partial \hat{I}}{\partial t} - \left(k_2 \hat{H} - \delta_2 \hat{I} - \mu \hat{I}^2\right)
$$

**Training**: Adam optimizer, lr = 1×10⁻³, 5000 epochs, λ = 1.0

---

## Project Structure

```
NeuroImmunoFly-SODE/
├── main.py                          # Master pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── README.md                        # This document
├── data/
│   ├── generate_connectome.py       # Tier-based connectome graph generator
│   ├── connectome_adj.npy           # (370, 370) adjacency matrix
│   ├── X_baseline.npy               # (1000, 370) synthetic baseline states
│   ├── sode_trajectories.csv        # SODE ground-truth trajectories
│   ├── spinn_predictions.csv        # S-PINN predicted trajectories
│   └── loss_history.npy             # Training loss log
├── models/
│   ├── sode_simulator.py            # Euler-Maruyama SODE solver
│   └── spinn_engine.py              # PyTorch S-PINN architecture & training
├── evaluation/
│   ├── compute_metrics.py           # RMSE, MAE, R², physics violations
│   ├── metrics_report.json          # Quantitative evaluation report
│   └── plot_generator.py            # Publication figure generator
└── outputs/
    ├── figures/
    │   ├── trajectory_cascade.png   # 3-panel true vs. predicted
    │   ├── phase_portrait.png       # H(t) vs I(t) phase space
    │   └── loss_convergence.png     # Training loss convergence
    └── weights/
        └── spinn_model.pth          # Trained S-PINN weights
```

---

## Execution Instructions

### Prerequisites

```bash
# Python 3.10+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Full Pipeline Execution

```bash
# Run the complete pipeline (Phases 1-5)
python main.py
```

### Individual Phase Execution

```bash
# Phase 1: Generate connectome graph
python data/generate_connectome.py

# Phase 2: Run SODE simulation
python models/sode_simulator.py

# Phase 3: Train S-PINN
python models/spinn_engine.py

# Phase 4: Compute metrics & generate figures
python evaluation/compute_metrics.py
python evaluation/plot_generator.py
```

---

## Output Description

### Figures

1. **`trajectory_cascade.png`**: Three-panel subplot comparing SODE ground-truth against S-PINN predictions for N(t), H(t), and I(t). The stress onset at t=20 is annotated with a vertical dashed line.

2. **`phase_portrait.png`**: 2D phase space trajectory of H(t) vs I(t), color-coded by time, showing convergence to the homeostatic attractor. The quadratic dampening term −μI² creates a stable fixed point in the H-I plane.

3. **`loss_convergence.png`**: Semi-logarithmic plot of data loss (L_data), physics residual loss (L_physics), and total loss (L_total) over 5000 training epochs.

### Metrics Report

The `metrics_report.json` contains:
- Per-compartment RMSE, MAE, and R²
- Physics violation residuals (mean and max per equation)
- Global RMSE across all compartments

---

## Biological Context

The Drosophila neuro-immuno-endocrine axis is organized around the **pars intercerebralis** (PI), a cluster of ~20 median neurosecretory cells homologous to the mammalian hypothalamic paraventricular nucleus. These neurons:

1. **Receive multimodal sensory input** via mushroom body output neurons (MBONs), integrating olfactory, gustatory, and mechanosensory signals
2. **Produce DH44** (diuretic hormone 44), functionally homologous to mammalian CRH, which drives downstream hormonal cascades
3. **Modulate systemic immunity** through insulin-like peptides (DILPs) and downstream signaling to hemocytes and fat body cells

The polyadic chemical synapses from sensory pathways to PI neurons represent the neuronal substrate of the stress response, where a single presynaptic terminal can contact multiple postsynaptic partners simultaneously — increasing the gain and reliability of stress signal propagation.

---

## References

1. Schlegel, P. et al. (2023). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *bioRxiv*.
2. Raissi, M. et al. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
3. Nässel, D.R. & Zandawala, M. (2019). Recent advances in neuropeptide signaling in *Drosophila*, from genes to physiology and behavior. *Progress in Neurobiology*, 179, 101607.
4. Hückesfeld, S. et al. (2021). Localization of motor neurons and central pattern generators for motor patterns underlying feeding behavior in *Drosophila* larvae. *PLoS ONE*, 16(3).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
