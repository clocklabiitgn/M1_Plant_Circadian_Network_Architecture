#!/usr/bin/env python3
"""
Build weighted Cytoscape-ready networks from:
  - combined_ll.xlsx  (LL condition)
  - combined_ld.xlsx  (LD condition)
  - Final_Parameter_Table_Updated.csv  (maps parameter -> source/target)

Outputs (in --outdir):
  edges_LL_period.csv
  edges_LL_geometry.csv
  edges_LL_composite.csv
  edges_LD_period.csv
  edges_LD_geometry.csv
  edges_LD_composite.csv
  edges_diff_LD_minus_LL_composite.csv

  nodes_LL_composite.csv
  nodes_LD_composite.csv

  network_LL_composite.pdf / .png
  network_LD_composite.pdf / .png
  network_diff_LD_minus_LL_composite.pdf / .png

Notes:
- "source" and "target" columns are used directly from the parameter table.
- "interaction" is taken directly (or inferred from parameter_description if missing).
- Degradation/translation/self-process parameters will often map to self-loops.

Run:
  python build_weighted_networks.py \
    --ll /path/combined_ll.xlsx \
    --ld /path/combined_ld.xlsx \
    --map /path/Final_Parameter_Table_Updated.csv \
    --outdir network_out
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# -----------------------------
# Config / helpers
# -----------------------------
REQUIRED_COMBINED_COLS = {
    "Name",
    "Knockout",
    "Mean_Delta_Period_knockout",
    "slope_mean_delta_period",
    "slope_mean_delta_area",
    "slope_mean_delta_eccentricity",
}

KNOCKOUT_CLASS_MAP = {
    "Class I": 1.0,
    "Class II": 0.5,
    "Class III": 0.0,
}

def safe_abs_norm(s: pd.Series) -> pd.Series:
    """Normalize by max(|x|). Returns 0 if all values are 0/NaN."""
    x = s.astype(float).replace([np.inf, -np.inf], np.nan)
    a = x.abs()
    m = np.nanmax(a.values) if np.isfinite(np.nanmax(a.values)) else np.nan
    if not np.isfinite(m) or m == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return a / m

def parse_interaction(desc: str) -> str:
    """Return 'inhibits', 'activates', or 'unknown' from description."""
    if not isinstance(desc, str):
        return "unknown"
    d = desc.lower()
    if "inhibition" in d or "inhibit" in d or "repress" in d:
        return "inhibits"
    if "activation" in d or "activate" in d:
        return "activates"
    return "unknown"

def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "savefig.bbox": "tight",
    })

def read_combined_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    missing = REQUIRED_COMBINED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df.copy()

def read_param_map_csv(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    needed = {"source", "target", "interaction", "parameter", "parameter_description"}
    missing = needed - set(m.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    m = m.copy()
    m["source"] = m["source"].astype(str).str.strip()
    m["target"] = m["target"].astype(str).str.strip()
    m["interaction"] = m["interaction"].astype(str).str.strip()
    m["parameter"] = m["parameter"].astype(str).str.strip()
    m["parameter_description"] = m["parameter_description"].astype(str)
    # infer interaction if missing/blank/"nan"
    missing_inter = m["interaction"].isna() | m["interaction"].str.len().eq(0) | m["interaction"].str.lower().eq("nan")
    m.loc[missing_inter, "interaction"] = m.loc[missing_inter, "parameter_description"].apply(parse_interaction)
    # keep one row per parameter (if duplicates exist, keep first)
    m = m.drop_duplicates(subset=["parameter"], keep="first")
    return m


# -----------------------------
# Weight construction
# -----------------------------
def build_weights(df: pd.DataFrame,
                  alpha_k: float,
                  beta_p: float,
                  gamma_a: float,
                  delta_e: float) -> pd.DataFrame:
    """
    Create normalized weights (0..1) and composite.
    K uses BOTH knockout class (structural) and knockout mean delta period (magnitude).
    """
    out = df.copy()

    # knockout class numeric
    out["knockout_class_num"] = out["Knockout"].map(KNOCKOUT_CLASS_MAP).fillna(0.0)

    # normalize metrics separately (within this condition)
    out["K_mag_norm"] = safe_abs_norm(out["Mean_Delta_Period_knockout"])
    out["P_norm"] = safe_abs_norm(out["slope_mean_delta_period"])
    out["A_norm"] = safe_abs_norm(out["slope_mean_delta_area"])
    out["E_norm"] = safe_abs_norm(out["slope_mean_delta_eccentricity"])

    # structural knockout score combines class + magnitude
    # (Class III becomes 0; Class I stays high if magnitude is high)
    out["K_norm"] = out["knockout_class_num"] * out["K_mag_norm"]

    # geometry: you can keep area & ecc separate, and also a combined geometry weight
    out["G_norm"] = 0.5 * (out["A_norm"] + out["E_norm"])

    # composite (weighted sum, rescaled to 0..1 by sum of coefficients)
    denom = (alpha_k + beta_p + gamma_a + delta_e)
    if denom == 0:
        raise ValueError("alpha_k + beta_p + gamma_a + delta_e must be > 0")
    out["W_composite"] = (
        alpha_k * out["K_norm"] +
        beta_p  * out["P_norm"] +
        gamma_a * out["A_norm"] +
        delta_e * out["E_norm"]
    ) / denom

    return out


# -----------------------------
# Cytoscape tables + analytics
# -----------------------------
def make_edge_table(weighted_df: pd.DataFrame, param_map: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    Join weights to parameter->edge mapping.
    Returns Cytoscape edge table (one row per Parameter).
    """
    j = weighted_df.merge(param_map[["parameter", "source", "target", "interaction", "parameter_description"]],
                          left_on="Name", right_on="parameter", how="left")

    # Mark unmapped parameters
    j["mapped"] = j["source"].notna() & j["target"].notna()

    # Fill missing mapping conservatively as self-loop on 'Component' parsed from Description if present.
    # If nothing is available, use parameter name as node.
    j["source"] = j["source"].fillna(j["Name"])
    j["target"] = j["target"].fillna(j["Name"])
    j["interaction"] = j["interaction"].fillna("unknown")

    edges = pd.DataFrame({
        "source": j["source"].astype(str),
        "target": j["target"].astype(str),
        "interaction": j["interaction"].astype(str),
        "parameter": j["Name"].astype(str),
        "parameter_description": j["Description"].astype(str) if "Description" in j.columns else j["parameter_description"].astype(str),
        "knockout_class": j["Knockout"].astype(str),
        "K_norm": j["K_norm"].astype(float),
        "P_norm": j["P_norm"].astype(float),
        "A_norm": j["A_norm"].astype(float),
        "E_norm": j["E_norm"].astype(float),
        "G_norm": j["G_norm"].astype(float),
        "W_composite": j["W_composite"].astype(float),
        "condition": condition,
        "mapped_from_table": j["mapped"].astype(bool),
    })

    return edges


def build_nx_graph(edges: pd.DataFrame, weight_col: str) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        w = float(r[weight_col])
        # store all weights for Cytoscape styling if needed
        G.add_edge(
            r["source"], r["target"],
            weight=w,
            interaction=r.get("interaction", "unknown"),
            parameter=r.get("parameter", ""),
            condition=r.get("condition", ""),
        )
    return G

def node_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute node strengths and centralities on a weighted directed graph.
    - strength_in/out: sum of edge weights
    - strength_total: sum of in+out
    - betweenness: weighted (distance = 1/weight)
    - eigenvector: computed on undirected projection (stable for most networks)
    """
    nodes = list(G.nodes())

    strength_in = {n: 0.0 for n in nodes}
    strength_out = {n: 0.0 for n in nodes}
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        strength_out[u] += abs(w)
        strength_in[v] += abs(w)

    strength_total = {n: strength_in[n] + strength_out[n] for n in nodes}

    # weighted betweenness: convert weight to distance (higher weight = shorter distance)
    H = G.copy()
    for u, v, d in H.edges(data=True):
        w = float(d.get("weight", 0.0))
        d["distance"] = (1.0 / w) if w > 0 else 1e9

    btw = nx.betweenness_centrality(H, weight="distance", normalized=True)

    # eigenvector on undirected
    U = H.to_undirected()
    try:
        eig = nx.eigenvector_centrality_numpy(U, weight="weight")
    except Exception:
        eig = {n: np.nan for n in nodes}

    df = pd.DataFrame({
        "node": nodes,
        "strength_in": [strength_in[n] for n in nodes],
        "strength_out": [strength_out[n] for n in nodes],
        "strength_total": [strength_total[n] for n in nodes],
        "betweenness": [btw.get(n, np.nan) for n in nodes],
        "eigenvector": [eig.get(n, np.nan) for n in nodes],
    }).sort_values("strength_total", ascending=False)

    return df


# -----------------------------
# Publication-ready plotting
# -----------------------------
def _bin_influence(values: dict[str, float]) -> dict[str, int]:
    """Bin node influence into 4 categories (1..4) using quartiles."""
    if not values:
        return {}
    v = np.array(list(values.values()), dtype=float)
    if np.allclose(v, v[0]):
        return {k: 2 for k in values}  # all same -> moderate
    q1, q2, q3 = np.nanpercentile(v, [25, 50, 75])
    binned = {}
    for k, val in values.items():
        if val <= q1:
            binned[k] = 1
        elif val <= q2:
            binned[k] = 2
        elif val <= q3:
            binned[k] = 3
        else:
            binned[k] = 4
    return binned

def _draw_self_loops(ax, pos, nodes, widths, colors, alpha=0.85) -> None:
    """Draw self-loops as small arcs around nodes."""
    for n, w, c in zip(nodes, widths, colors):
        x, y = pos[n]
        loop = FancyArrowPatch(
            (x, y), (x, y),
            connectionstyle="arc3,rad=0.5",
            arrowstyle="-|>",
            mutation_scale=10,
            lw=w,
            color=c,
            alpha=alpha,
        )
        ax.add_patch(loop)

def draw_network(G: nx.DiGraph, node_df: pd.DataFrame, out_pdf: Path, out_png: Path, title: str) -> None:
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 5.0), dpi=300)
    ax = plt.gca()
    ax.set_title(title)

    # deterministic layout (circular for clean, symmetric look)
    pos = nx.circular_layout(G, scale=1.0)

    # node sizes based on strength_total
    st = node_df.set_index("node")["strength_total"].to_dict()
    node_sizes = []
    for n in G.nodes():
        s = float(st.get(n, 0.0))
        # scale to reasonable point area
        node_sizes.append(200 + 2500 * (s / (max(st.values()) if len(st) and max(st.values()) > 0 else 1.0)))

    # node colors by binned influence
    influence_bins = _bin_influence(st)
    node_palette = {
        1: "#b9dceb",  # low
        2: "#f09a9a",  # moderate
        3: "#d35d5d",  # high
        4: "#9e1f1f",  # very high
    }
    node_colors = [node_palette.get(influence_bins.get(n, 2), "#f09a9a") for n in G.nodes()]

    # edge widths based on weight
    weights = [float(d.get("weight", 0.0)) for _, _, d in G.edges(data=True)]
    maxw = max(weights) if weights else 1.0
    edge_widths = [0.5 + 4.0 * (w / maxw) for w in weights]

    # edge colors by interaction
    edge_colors = []
    for _, _, d in G.edges(data=True):
        inter = str(d.get("interaction", "unknown")).lower()
        if "inhibit" in inter or "repress" in inter:
            edge_colors.append("#8ecae6")  # blue
        elif "activate" in inter:
            edge_colors.append("#f26c6c")  # red
        else:
            edge_colors.append("#9aa0a6")  # grey

    # separate self-loops for cleaner rendering
    self_loops = [(u, v, d, w, c) for (u, v, d), w, c in zip(G.edges(data=True), edge_widths, edge_colors) if u == v]
    non_self = [(u, v, d, w, c) for (u, v, d), w, c in zip(G.edges(data=True), edge_widths, edge_colors) if u != v]

    # draw non-self edges with gentle curvature
    if non_self:
        edges = [(u, v) for u, v, _, _, _ in non_self]
        widths = [w for _, _, _, w, _ in non_self]
        colors = [c for _, _, _, _, c in non_self]
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges,
            width=widths,
            edge_color=colors,
            arrows=True,
            arrowsize=10,
            alpha=0.85,
            connectionstyle="arc3,rad=0.18",
        )

    # draw self-loops
    if self_loops:
        loop_nodes = [u for u, _, _, _, _ in self_loops]
        loop_widths = [w for _, _, _, w, _ in self_loops]
        loop_colors = [c for _, _, _, _, c in self_loops]
        _draw_self_loops(ax, pos, loop_nodes, loop_widths, loop_colors, alpha=0.85)

    # draw nodes + labels
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, linewidths=1.0, edgecolors="#333333")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # legend (node influence bins)
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=node_palette[1], edgecolor="#333333", label="Low Influence"),
        Patch(facecolor=node_palette[2], edgecolor="#333333", label="Moderate Influence"),
        Patch(facecolor=node_palette[3], edgecolor="#333333", label="High Influence"),
        Patch(facecolor=node_palette[4], edgecolor="#333333", label="Very High Influence"),
    ]
    ax.legend(handles=legend_items, title="Node Influence", loc="lower left", frameon=True)

    ax.axis("off")
    fig.tight_layout()

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ll", type=Path, required=True, help="combined_ll.xlsx")
    ap.add_argument("--ld", type=Path, required=True, help="combined_ld.xlsx")
    ap.add_argument("--map", type=Path, required=True, help="Final_Parameter_Table_Updated.csv")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory")

    # composite coefficients
    ap.add_argument("--alpha_k", type=float, default=1.0, help="Weight for knockout term")
    ap.add_argument("--beta_p", type=float, default=1.0, help="Weight for period sensitivity term")
    ap.add_argument("--gamma_a", type=float, default=1.0, help="Weight for area slope term")
    ap.add_argument("--delta_e", type=float, default=1.0, help="Weight for eccentricity slope term")

    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # load
    ll_df = read_combined_xlsx(args.ll)
    ld_df = read_combined_xlsx(args.ld)
    pmap = read_param_map_csv(args.map)

    # weights per condition (normalized separately)
    ll_w = build_weights(ll_df, args.alpha_k, args.beta_p, args.gamma_a, args.delta_e)
    ld_w = build_weights(ld_df, args.alpha_k, args.beta_p, args.gamma_a, args.delta_e)

    # edge tables
    edges_ll = make_edge_table(ll_w, pmap, "LL")
    edges_ld = make_edge_table(ld_w, pmap, "LD")

    # save Cytoscape-ready edge files for each network type
    edges_ll_period = edges_ll.copy()
    edges_ll_period["weight"] = edges_ll_period["P_norm"]
    edges_ll_period.to_csv(outdir / "edges_LL_period.csv", index=False)

    edges_ll_geom = edges_ll.copy()
    edges_ll_geom["weight"] = edges_ll_geom["G_norm"]
    edges_ll_geom.to_csv(outdir / "edges_LL_geometry.csv", index=False)

    edges_ll_comp = edges_ll.copy()
    edges_ll_comp["weight"] = edges_ll_comp["W_composite"]
    edges_ll_comp.to_csv(outdir / "edges_LL_composite.csv", index=False)

    edges_ld_period = edges_ld.copy()
    edges_ld_period["weight"] = edges_ld_period["P_norm"]
    edges_ld_period.to_csv(outdir / "edges_LD_period.csv", index=False)

    edges_ld_geom = edges_ld.copy()
    edges_ld_geom["weight"] = edges_ld_geom["G_norm"]
    edges_ld_geom.to_csv(outdir / "edges_LD_geometry.csv", index=False)

    edges_ld_comp = edges_ld.copy()
    edges_ld_comp["weight"] = edges_ld_comp["W_composite"]
    edges_ld_comp.to_csv(outdir / "edges_LD_composite.csv", index=False)

    # differential composite: (LD - LL) by parameter (same mapping)
    diff = edges_ld_comp.merge(
        edges_ll_comp[["parameter", "source", "target", "interaction", "weight"]],
        on=["parameter", "source", "target", "interaction"],
        how="inner",
        suffixes=("_LD", "_LL"),
    )
    diff["weight_diff_LD_minus_LL"] = diff["weight_LD"] - diff["weight_LL"]
    diff_out = diff[[
        "source", "target", "interaction", "parameter",
        "weight_LL", "weight_LD", "weight_diff_LD_minus_LL"
    ]].copy()
    diff_out.to_csv(outdir / "edges_diff_LD_minus_LL_composite.csv", index=False)

    # Node metrics + plots for composite networks
    G_ll = build_nx_graph(edges_ll_comp.rename(columns={"weight": "W"}), "W_composite")
    nodes_ll = node_metrics(G_ll)
    nodes_ll.to_csv(outdir / "nodes_LL_composite.csv", index=False)

    G_ld = build_nx_graph(edges_ld_comp.rename(columns={"weight": "W"}), "W_composite")
    nodes_ld = node_metrics(G_ld)
    nodes_ld.to_csv(outdir / "nodes_LD_composite.csv", index=False)

    # Diff graph uses absolute value for plotting size; sign is kept in edge attribute
    G_diff = nx.DiGraph()
    for _, r in diff_out.iterrows():
        w = float(r["weight_diff_LD_minus_LL"])
        G_diff.add_edge(r["source"], r["target"], weight=abs(w), signed_weight=w, interaction=r["interaction"], parameter=r["parameter"])
    nodes_diff = node_metrics(G_diff)
    nodes_diff.to_csv(outdir / "nodes_diff_LD_minus_LL_composite.csv", index=False)

    # write GraphML for Cytoscape (layout + styling can be adjusted in Cytoscape)
    nx.write_graphml(G_ll, outdir / "network_LL_composite.graphml")
    nx.write_graphml(G_ld, outdir / "network_LD_composite.graphml")
    nx.write_graphml(G_diff, outdir / "network_diff_LD_minus_LL_composite.graphml")

    # publication-ready images
    draw_network(G_ll, nodes_ll, outdir / "network_LL_composite.pdf", outdir / "network_LL_composite.png", "Composite network (LL)")
    draw_network(G_ld, nodes_ld, outdir / "network_LD_composite.pdf", outdir / "network_LD_composite.png", "Composite network (LD)")
    draw_network(G_diff, nodes_diff, outdir / "network_diff_LD_minus_LL_composite.pdf", outdir / "network_diff_LD_minus_LL_composite.png",
                 "Differential composite network (LD − LL)")

    # Also save adjacency matrices (composite) for LL and LD
    def adjacency_csv(G: nx.DiGraph, path: Path) -> None:
        nodes = sorted(G.nodes())
        A = pd.DataFrame(0.0, index=nodes, columns=nodes)
        for u, v, d in G.edges(data=True):
            A.loc[u, v] = float(d.get("weight", 0.0))
        A.to_csv(path)

    adjacency_csv(G_ll, outdir / "adjacency_LL_composite.csv")
    adjacency_csv(G_ld, outdir / "adjacency_LD_composite.csv")
    adjacency_csv(G_diff, outdir / "adjacency_diff_LD_minus_LL_composite_abs.csv")

    print(f"\nDone. Outputs written to: {outdir.resolve()}")
    print("Cytoscape import tips:")
    print("  1) File → Import → Network from File… (use edges_*.csv)")
    print("  2) Ensure columns map as: Source=source, Target=target, Interaction=interaction")
    print("  3) Then import nodes_*.csv via: File → Import → Table from File… (Key column = node)")
    print("  4) Style: map edge width to 'weight' (or W_composite), node size to 'strength_total'\n")


if __name__ == "__main__":
    main()
