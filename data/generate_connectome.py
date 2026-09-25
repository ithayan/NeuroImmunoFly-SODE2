"""
Connectome Graph Generator for NeuroImmunoFly-SODE
===================================================
Builds a directed multi-tier neuronal connectivity graph from the
Janelia FlyEM Male CNS (malecns) connectomics dataset.

Data Source:
  malecns-master/data-raw/2023-27-2 soma_sides.csv
  (125,506 neurons from the whole male Drosophila CNS)

The dataset contains soma coordinates (nx, ny, nz), body IDs,
presynaptic T-bar counts (tbars), body sizes, and soma laterality.

Tier Assignment Strategy:
  Neurons are assigned to tiers based on anatomical position (z-depth)
  and functional properties (tbars, body_size) reflecting the real
  Drosophila CNS cytoarchitecture:

  T1 (Sensory/Inputs)             — anterior/peripheral neurons with
                                    small bodies (optic lobe, antennal)
  T2 (Interneurons/Mushroom Body) — large-body, high-tbar central
                                    processing neurons
  T3 (Neurosecretory / PI cells)  — medial neurons in the pars
                                    intercerebralis region (dorsomedial)
  T4 (Systemic Immune Effectors)  — posterior descending neurons and
                                    VNC-projecting efferents

Outputs:
  data/connectome_adj.npy   — (N_selected, N_selected) weighted adjacency
  data/X_baseline.npy       — (1000, N_selected) synthetic baseline states
  data/malecns_metadata.csv — tier assignments for selected neurons
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# --- tier sizes (sampled from the full 125k neurons) ---
TIER_CONFIG = {
    "T1_sensory":        {"n": 50,  "label": "Sensory/Input"},
    "T2_mushroom_body":  {"n": 200, "label": "Interneuron/MB"},
    "T3_neurosecretory": {"n": 20,  "label": "Neurosecretory/PI"},
    "T4_immune":         {"n": 100, "label": "Immune Effector"},
}
TOTAL_NODES = sum(t["n"] for t in TIER_CONFIG.values())  # 370

# Connection probabilities between tiers (feedforward + feedback)
INTER_TIER_P = {
    ("T1_sensory", "T2_mushroom_body"):        0.08,
    ("T2_mushroom_body", "T3_neurosecretory"):  0.12,
    ("T3_neurosecretory", "T4_immune"):         0.15,
    ("T2_mushroom_body", "T1_sensory"):         0.02,   # feedback
    ("T4_immune", "T3_neurosecretory"):         0.05,   # immune feedback
}
INTRA_TIER_P = 0.03  # lateral connectivity within a tier


def load_malecns_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the Male CNS soma dataset.

    Parameters
    ----------
    data_path : str
        Path to '2023-27-2 soma_sides.csv'

    Returns
    -------
    df : DataFrame
        Cleaned dataset with columns: body, nx, ny, nz, tbars, body_size, soma_side
    """
    df = pd.read_csv(data_path)
    df = df.rename(columns={"Unnamed: 0": "idx"})

    # Drop neurons with no body or zero size
    df = df[df["body"] > 0].copy()
    df = df[df["body_size"] > 0].copy()

    # Normalize coordinates to [0, 1] for tier assignment
    for col in ["nx", "ny", "nz"]:
        df[f"{col}_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)

    print(f"  Loaded {len(df)} neurons from Male CNS dataset")
    print(f"  Coordinate ranges:")
    print(f"    x: [{df['nx'].min()}, {df['nx'].max()}]")
    print(f"    y: [{df['ny'].min()}, {df['ny'].max()}]")
    print(f"    z: [{df['nz'].min()}, {df['nz'].max()}]")
    print(f"  T-bar range: [{df['tbars'].min()}, {df['tbars'].max()}]")
    print(f"  Laterality: L={len(df[df.soma_side=='L'])}, R={len(df[df.soma_side=='R'])}, M={len(df[df.soma_side=='M'])}")

    return df


def assign_tiers(df: pd.DataFrame) -> dict:
    """
    Assign neurons to functional tiers based on anatomical position
    and synaptic properties from the real Male CNS data.

    Tier assignment heuristics (informed by Drosophila neuroanatomy):
      T1 (Sensory): Anterior neurons (low nz_norm) with moderate tbars
                    -> optic lobe, antennal sensory pathways
      T2 (MB/Interneurons): Central neurons with high tbars and large body_size
                    -> mushroom body Kenyon cells, MB output neurons
      T3 (Neurosecretory/PI): Dorsomedial neurons (high ny_norm, medial nx)
                    -> pars intercerebralis CRH-homolog neurons
      T4 (Immune Effectors): Posterior neurons (high nz_norm) with moderate tbars
                    -> descending neurons, VNC-projecting immune modulators
    """
    selected = {}

    # --- T1: Sensory/anterior neurons ---
    # Low z (anterior), moderate tbars (sensory processing)
    t1_candidates = df[
        (df["nz_norm"] < 0.35) &
        (df["tbars"] > 0) &
        (df["tbars"] < df["tbars"].quantile(0.7))
    ].copy()
    t1_candidates["score"] = (1 - df["nz_norm"]) * 0.7 + df["nx_norm"].apply(lambda x: 1 - abs(x - 0.5)) * 0.3
    t1_sample = t1_candidates.nlargest(TIER_CONFIG["T1_sensory"]["n"] * 3, "score")
    t1 = t1_sample.sample(n=TIER_CONFIG["T1_sensory"]["n"], random_state=SEED)
    selected["T1_sensory"] = t1
    used_bodies = set(t1["body"].values)

    # --- T2: Mushroom Body / Interneurons ---
    # Central, high tbars, large body
    t2_candidates = df[
        ~df["body"].isin(used_bodies) &
        (df["nz_norm"].between(0.25, 0.75)) &
        (df["tbars"] > df["tbars"].quantile(0.5)) &
        (df["body_size"] > df["body_size"].quantile(0.3))
    ].copy()
    t2_candidates["score"] = (
        df["tbars"] / df["tbars"].max() * 0.4 +
        df["body_size"].apply(np.log1p) / df["body_size"].apply(np.log1p).max() * 0.3 +
        (1 - abs(df["nz_norm"] - 0.5)) * 0.3
    )
    t2_sample = t2_candidates.nlargest(TIER_CONFIG["T2_mushroom_body"]["n"] * 3, "score")
    t2 = t2_sample.sample(n=TIER_CONFIG["T2_mushroom_body"]["n"], random_state=SEED)
    selected["T2_mushroom_body"] = t2
    used_bodies.update(t2["body"].values)

    # --- T3: Neurosecretory / Pars Intercerebralis ---
    # Dorsomedial: high ny_norm (dorsal), medial nx (close to midline)
    # The PI is a bilaterally symmetric cluster near the midline
    midline_x = df["nx_norm"].median()
    t3_candidates = df[
        ~df["body"].isin(used_bodies) &
        (df["ny_norm"] > 0.55) &
        (abs(df["nx_norm"] - midline_x) < 0.2) &
        (df["nz_norm"].between(0.3, 0.65))
    ].copy()
    t3_candidates["score"] = (
        df["ny_norm"] * 0.4 +
        (1 - abs(df["nx_norm"] - midline_x)) * 0.4 +
        df["body_size"].apply(np.log1p) / df["body_size"].apply(np.log1p).max() * 0.2
    )
    t3_sample = t3_candidates.nlargest(TIER_CONFIG["T3_neurosecretory"]["n"] * 5, "score")
    t3 = t3_sample.sample(n=min(TIER_CONFIG["T3_neurosecretory"]["n"], len(t3_sample)), random_state=SEED)
    # If not enough, fill from remaining medial neurons
    if len(t3) < TIER_CONFIG["T3_neurosecretory"]["n"]:
        deficit = TIER_CONFIG["T3_neurosecretory"]["n"] - len(t3)
        extra = df[
            ~df["body"].isin(used_bodies | set(t3["body"].values)) &
            (abs(df["nx_norm"] - midline_x) < 0.25)
        ].sample(n=deficit, random_state=SEED+1)
        t3 = pd.concat([t3, extra])
    selected["T3_neurosecretory"] = t3
    used_bodies.update(t3["body"].values)

    # --- T4: Immune Effectors / Posterior descending ---
    # Posterior (high nz_norm), moderate body size
    t4_candidates = df[
        ~df["body"].isin(used_bodies) &
        (df["nz_norm"] > 0.55) &
        (df["body_size"] > df["body_size"].quantile(0.2))
    ].copy()
    t4_candidates["score"] = (
        df["nz_norm"] * 0.5 +
        df["body_size"].apply(np.log1p) / df["body_size"].apply(np.log1p).max() * 0.3 +
        df["tbars"] / (df["tbars"].max() + 1) * 0.2
    )
    t4_sample = t4_candidates.nlargest(TIER_CONFIG["T4_immune"]["n"] * 3, "score")
    t4 = t4_sample.sample(n=TIER_CONFIG["T4_immune"]["n"], random_state=SEED)
    selected["T4_immune"] = t4
    used_bodies.update(t4["body"].values)

    return selected


def build_connectome_graph(tier_data: dict) -> nx.DiGraph:
    """
    Construct a directed graph with biologically-motivated stochastic
    connectivity between and within tiers, using real neuron properties
    as edge weights.

    Edge weight = f(source_tbars, distance) -- neurons with more
    presynaptic T-bars form stronger connections, attenuated by distance.
    """
    G = nx.DiGraph()

    # --- add nodes with real metadata ---
    node_id = 0
    tier_node_map = {}  # tier_name -> list of node_ids
    node_metadata = []

    for tier_name, tier_df in tier_data.items():
        tier_nodes = []
        for _, row in tier_df.iterrows():
            G.add_node(node_id,
                       tier=tier_name,
                       body_id=int(row["body"]),
                       nx=float(row["nx"]),
                       ny=float(row["ny"]),
                       nz=float(row["nz"]),
                       tbars=int(row["tbars"]),
                       body_size=int(row["body_size"]),
                       soma_side=str(row["soma_side"]))

            node_metadata.append({
                "node_id": node_id,
                "tier": tier_name,
                "body_id": int(row["body"]),
                "nx": float(row["nx"]),
                "ny": float(row["ny"]),
                "nz": float(row["nz"]),
                "tbars": int(row["tbars"]),
                "body_size": int(row["body_size"]),
                "soma_side": str(row["soma_side"]),
            })

            tier_nodes.append(node_id)
            node_id += 1

        tier_node_map[tier_name] = tier_nodes

    # --- inter-tier edges (feedforward + feedback) ---
    for (src_tier, dst_tier), p in INTER_TIER_P.items():
        src_nodes = tier_node_map[src_tier]
        dst_nodes = tier_node_map[dst_tier]
        for s in src_nodes:
            src_data = G.nodes[s]
            for d in dst_nodes:
                if rng.random() < p:
                    dst_data = G.nodes[d]
                    # Distance-weighted connectivity
                    dist = np.sqrt(
                        (src_data["nx"] - dst_data["nx"])**2 +
                        (src_data["ny"] - dst_data["ny"])**2 +
                        (src_data["nz"] - dst_data["nz"])**2
                    )
                    # Weight: tbar-based strength * distance decay
                    w = (src_data["tbars"] + 1) / (dist + 1000)
                    w = float(np.clip(w, 0.01, 2.0))
                    G.add_edge(s, d, weight=w)

    # --- intra-tier lateral connectivity ---
    for tier_name, nodes in tier_node_map.items():
        for i, s in enumerate(nodes):
            for d in nodes[i + 1:]:
                if rng.random() < INTRA_TIER_P:
                    src_data = G.nodes[s]
                    dst_data = G.nodes[d]
                    dist = np.sqrt(
                        (src_data["nx"] - dst_data["nx"])**2 +
                        (src_data["ny"] - dst_data["ny"])**2 +
                        (src_data["nz"] - dst_data["nz"])**2
                    )
                    w = float(np.clip(rng.uniform(0.05, 0.5) * 1000 / (dist + 500), 0.01, 1.0))
                    G.add_edge(s, d, weight=w)
                if rng.random() < INTRA_TIER_P:
                    G.add_edge(d, s, weight=rng.uniform(0.05, 0.3))

    return G, pd.DataFrame(node_metadata)


def generate_adjacency_matrix(G: nx.DiGraph) -> np.ndarray:
    """Return a dense (N, N) weighted adjacency matrix."""
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()), weight="weight")
    return A.astype(np.float32)


def generate_baseline_states(n_samples: int = 1000, n_nodes: int = None) -> np.ndarray:
    """
    Generate synthetic baseline state vectors X in R^{n_samples x n_nodes}.
    Each entry is a continuous value in [0, 1) representing normalized
    membrane potentials / molecular concentrations.
    """
    return rng.uniform(0.0, 1.0, size=(n_samples, n_nodes)).astype(np.float32)


def run(output_dir: str = None):
    """Main entry point for connectome generation from Male CNS data."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the malecns dataset
    malecns_csv = os.path.join(output_dir, "malecns", "malecns-master",
                                "data-raw", "2023-27-2 soma_sides.csv")

    if not os.path.exists(malecns_csv):
        raise FileNotFoundError(
            f"Male CNS dataset not found at {malecns_csv}. "
            f"Please extract malecns-master.zip to data/malecns/"
        )

    print("[Phase 1] Building connectome from Male CNS dataset...")
    print(f"  Source: {malecns_csv}")

    # Load real data
    df = load_malecns_data(malecns_csv)

    # Assign neurons to functional tiers
    print("\n  Assigning neurons to functional tiers...")
    tier_data = assign_tiers(df)

    for tier_name, tier_df in tier_data.items():
        label = TIER_CONFIG[tier_name]["label"]
        print(f"    {tier_name:25s} ({label:25s}): {len(tier_df):>4d} neurons, "
              f"mean tbars={tier_df['tbars'].mean():.1f}, "
              f"mean body_size={tier_df['body_size'].mean():.0f}")

    # Build directed graph
    print("\n  Constructing directed connectivity graph...")
    G, metadata_df = build_connectome_graph(tier_data)

    print(f"    Nodes : {G.number_of_nodes()}")
    print(f"    Edges : {G.number_of_edges()}")

    # Per-tier edge stats
    for tier_name in TIER_CONFIG:
        tier_nodes = [n for n, d in G.nodes(data=True) if d.get("tier") == tier_name]
        outgoing = sum(1 for u, _ in G.edges() if u in tier_nodes)
        incoming = sum(1 for _, v in G.edges() if v in tier_nodes)
        print(f"    {tier_name:25s}: out={outgoing:>5d}, in={incoming:>5d}")

    # --- Save adjacency matrix ---
    A = generate_adjacency_matrix(G)
    adj_path = os.path.join(output_dir, "connectome_adj.npy")
    np.save(adj_path, A)
    print(f"\n  Adjacency matrix saved  -> {adj_path}  shape={A.shape}")
    print(f"  Density: {np.count_nonzero(A) / A.size:.4f}")
    print(f"  Weight range: [{A[A > 0].min():.4f}, {A[A > 0].max():.4f}]")

    # --- Save baseline states ---
    X = generate_baseline_states(n_nodes=TOTAL_NODES)
    base_path = os.path.join(output_dir, "X_baseline.npy")
    np.save(base_path, X)
    print(f"  Baseline states saved   -> {base_path}  shape={X.shape}")

    # --- Save metadata ---
    meta_path = os.path.join(output_dir, "malecns_metadata.csv")
    metadata_df.to_csv(meta_path, index=False)
    print(f"  Neuron metadata saved   -> {meta_path}  ({len(metadata_df)} neurons)")

    print("[Phase 1] Complete.\n")

    return G, A, X


if __name__ == "__main__":
    run()
