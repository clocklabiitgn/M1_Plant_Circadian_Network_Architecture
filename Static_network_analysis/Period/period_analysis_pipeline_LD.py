#!/usr/bin/env python3
"""
period_analysis_pipeline_LD.py

COMBINED PIPELINE — Period Sensitivity Analysis | LD (Light/Dark 12:12) condition
==================================================================================
Step 1 — Load period_data_*.csv from parameter_analysis/ subfolders,
          compute % variation per component, generate per-parameter line plots,
          a sensitivity heatmap, and an influence ranking.

Step 2 — Build a period-delta table (each row's period minus the period at
          the base parameter value) and compute fold-change vs base.

Step 3 — Plot period vs fold-change curves, per-component slope bar charts,
          a stacked 4-panel slope figure, CV computation, and a combined
          mean-slope bar plot.  Save all CSVs and PNGs.

All outputs land in:  <script_dir>/LD_period_outputs/

Run:
  python period_analysis_pipeline_LD.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — LD condition
# ──────────────────────────────────────────────────────────────────────────────
CONDITION_LABEL  = "LD"
OUTPUT_DIR_NAME  = "LD_period_outputs"

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "parameter_analysis_ld"
OUT_DIR   = BASE_DIR / OUTPUT_DIR_NAME

COMPONENTS = ["LHYm", "P97m", "P51m", "ELm"]

# Base parameter values
BASE_PARAMS: dict[str, float] = {
    # ── CL (LHY / CCA1) module ────────────────────────────────────────────────
    "v1":    4.8318,   # CL basal synthesis
    "q1a":   1.4266,   # CL light induction via PhyA
    "q3a":   8.9432,   # CL light induction via PhyB
    "q4a":   5.9277,   # CL light induction via Cry1
    "K1":    0.1943,   # Inhibition of CL by P97
    "K2":    1.6138,   # Inhibition of CL by P51
    "k1L":   0.2866,   # CL mRNA degradation (light)
    "k1D":   0.213,    # CL mRNA degradation (dark)
    "p1":    0.8672,   # CL translation
    "p1L":   0.2378,   # CL light-induced translation
    "d1":    0.7843,   # CL protein degradation
    # ── P97 (PRR9 / PRR7) module ──────────────────────────────────────────────
    "q1b":   3.575,    # P97 light induction via PhyA
    "q3b":   5.5899,   # P97 light induction via PhyB
    "q4b":   8.954,    # P97 light induction via Cry1
    "v2":    1.6822,   # P97 basal synthesis
    "K3":    2.2275,   # Inhibition of P97 by CL
    "K4":    0.40,     # Inhibition of P97 by P51
    "K5":    0.37,     # Inhibition of P97 by EL
    "k2":    0.35,     # P97 mRNA degradation
    "p2":    0.7858,   # P97 translation
    "d2D":   0.3712,   # P97 protein degradation (dark)
    "d2L":   0.2917,   # P97 protein degradation (light)
    # ── P51 (PRR5 / TOC1) module ──────────────────────────────────────────────
    "v3":    1.113,    # P51 basal synthesis
    "K6":    0.4944,   # Inhibition of P51 by CL
    "K7":    2.4087,   # Inhibition of P51 by P51 (self-repression)
    "k3":    0.5819,   # P51 mRNA degradation
    "p3":    0.6142,   # P51 translation
    "d3D":   0.5026,   # P51 protein degradation (dark)
    "d3L":   0.5431,   # P51 protein degradation (light)
    # ── EL (ELF4 / LUX) module ────────────────────────────────────────────────
    "v4":    2.5012,   # EL basal synthesis
    "K8":    0.3262,   # Inhibition of EL by CL
    "K9":    1.7974,   # Inhibition of EL by P51
    "K10":   1.1889,   # Inhibition of EL by EL protein
    "k4":    0.925,    # EL mRNA degradation
    "p4":    1.126,    # EL translation
    "de1":   0.0022,   # EL basal degradation
    "de2":   0.4741,   # EL degradation via free COP1
    "de3":   0.3765,   # EL degradation via COP1:PhyA
    "de4":   0.398,    # EL degradation via COP1:PhyB
    "de5":   0.0003,   # EL degradation via COP1:Cry1
    # ── PhyA module ───────────────────────────────────────────────────────────
    "Ap3":   0.3868,   # PhyA synthesis
    "Am7":   0.5503,   # PhyA Michaelis degradation Vmax
    "Ak7":   1.125,    # PhyA Michaelis constant
    "q2":    0.5767,   # Light-independent PhyA inactivation
    "kmpac": 137.0,    # COP1:PhyA binding rate
    "kd":    7.0,      # COP1 complex dissociation rate
    # ── PIF module ────────────────────────────────────────────────────────────
    "v5":    0.1129,   # PIF basal synthesis
    "K11":   0.3322,   # Inhibition of PIF by EL
    "K14":   1.5,      # Inhibition of PIF by Cry1
    "k5":    0.1591,   # PIF mRNA degradation
    "p5":    0.5293,   # PIF translation
    "d5D":   0.4404,   # PIF protein degradation (dark)
    "d5L":   5.0712,   # PIF protein degradation (light)
    # ── Hypocotyl ─────────────────────────────────────────────────────────────
    "g1":    0.001,    # Basal hypocotyl growth
    "g2":    0.18,     # PIF-induced hypocotyl growth
    "K12":   0.86,     # Activation of growth by PIF
    # ── PhyB module ───────────────────────────────────────────────────────────
    "Bp4":   0.4147,   # PhyB synthesis
    "Bm8":   0.7728,   # PhyB Michaelis degradation Vmax
    "Bk8":   0.1732,   # PhyB Michaelis constant
    "kmpbc": 7162.0,   # COP1:PhyB binding rate
    # ── Cry1 module ───────────────────────────────────────────────────────────
    "Cp5":   0.4567,   # Cry1 synthesis
    "Cm9":   0.867,    # Cry1 Michaelis degradation Vmax
    "Ck9":   0.3237,   # Cry1 Michaelis constant
    "kmcc":  13406.0,  # COP1:Cry1 binding rate
    # ── GZ module (direct protein dynamics — no mRNA state) ───────────────────
    "Gp6":   0.000100, # GZ synthesis rate
    "dg1":   0.010000, # GZ basal degradation
    "dg2":   1.280202, # GZ degradation via free COP1
    "dg3":   0.010000, # GZ degradation via COP1:PhyA
    "dg4":   1.750462, # GZ degradation via COP1:PhyB
    "dg5":   1.067661, # GZ degradation via COP1:Cry1
    "dp6":   0.010000, # GZ-mediated P51 protein degradation rate
    "Gkp":   1.185527, # Michaelis constant for GZ-mediated P51 degradation
    # ── Light normalisation constants (fixed — not knocked out) ───────────────
    "eta1":  0.03,
    "eta2":  0.0215,
}

# Okabe–Ito palette (colorblind-friendly)
OKABE_ITO = [
    "#0072B2", "#E69F00", "#009E73", "#56B4E9",
    "#D55E00", "#CC79A7", "#F0E442", "#000000",
]

SLOPE_COLOR_MAP = {
    "Highly Sensitive":     "#D55E00",
    "Moderately Sensitive": "#E69F00",
    "Less Sensitive":       "#999999",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _sensitivity_level(pct: float) -> str:
    if pct > 20:
        return "Highly sensitive"
    elif pct > 5:
        return "Moderately sensitive"
    return "Insensitive"


def _ensure_out(*subdirs: str) -> Path:
    p = OUT_DIR.joinpath(*subdirs)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load, analyse, per-parameter line plots + heatmap
# ══════════════════════════════════════════════════════════════════════════════
def step1_load_and_analyse() -> pd.DataFrame:
    print("\n" + "="*60)
    print(f"[ Step 1 | {CONDITION_LABEL} ] Loading period data & sensitivity ...")
    print("="*60)

    _ensure_out()   # create OUT_DIR
    summary_data, heatmap_data, aggregated_frames = [], [], []

    for folder in sorted(DATA_DIR.glob("*/")):
        if not folder.is_dir():
            continue
        param_name = folder.name
        csv_files = list(folder.glob("period_data_*.csv"))
        if not csv_files:
            print(f"  ⚠  No period_data_*.csv in: {param_name}")
            continue

        df = pd.read_csv(csv_files[0])
        param_col = _find_col(df, ["param", "Parameter Value", "Parameter_Value",
                                   "parameter", "Parameter"])
        if param_col is None:
            print(f"  ⚠  Missing parameter column in {csv_files[0].name} — skipping")
            continue

        df_meta = df.copy()
        df_meta["Parameter"] = param_name
        df_meta["Source CSV"] = csv_files[0].name
        aggregated_frames.append(df_meta)

        component_stats: dict[str, str] = {}
        heatmap_row: dict[str, object] = {"Parameter": param_name}

        # Per-parameter output subfolder (mirrors original script 1 behaviour)
        param_plot_dir = _ensure_out("per_parameter_plots", param_name)

        for comp in COMPONENTS:
            if comp not in df.columns:
                component_stats[comp] = "Missing"
                heatmap_row[comp] = np.nan
                continue

            period_vals = df[comp].dropna()
            if period_vals.empty:
                component_stats[comp] = "No Rhythmicity"
                heatmap_row[comp] = np.nan
                continue

            pct_var = 100.0 * (period_vals.max() - period_vals.min()) / period_vals.mean()
            heatmap_row[comp] = pct_var
            component_stats[comp] = f"{_sensitivity_level(pct_var)} ({pct_var:.2f}%)"

            fig, ax = plt.subplots()
            ax.plot(df[param_col], df[comp], marker="o", linewidth=1.6,
                    color=OKABE_ITO[0])
            ax.set_title(f"[{CONDITION_LABEL}] {param_name} → {comp}")
            ax.set_xlabel(param_col)
            ax.set_ylabel("Period (h)")
            ax.grid(True, linestyle=":", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            plt.savefig(param_plot_dir / f"period_plot_{comp}.png", dpi=300)
            plt.close(fig)

        summary_data.append({"Parameter": param_name, **component_stats})
        heatmap_data.append(heatmap_row)

    if not aggregated_frames:
        raise RuntimeError(f"No valid period data found under: {DATA_DIR}")

    # Sensitivity summary
    pd.DataFrame(summary_data).to_csv(
        OUT_DIR / "parameter_period_sensitivity_summary.csv", index=False)

    # Heatmap
    heatmap_df = pd.DataFrame(heatmap_data).set_index("Parameter")
    fig, ax = plt.subplots(figsize=(10, max(6, len(heatmap_df) * 0.35)))
    sns.heatmap(heatmap_df[COMPONENTS], annot=True, fmt=".1f",
                cmap="coolwarm", cbar_kws={"label": "% Variation"},
                linewidths=0.4, ax=ax)
    ax.set_title(f"[{CONDITION_LABEL}] Period Sensitivity Heatmap (% variation)", pad=10)
    ax.set_ylabel("Parameter")
    ax.set_xlabel("Component")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "period_sensitivity_heatmap.png", dpi=300)
    plt.close(fig)

    # Influence ranking
    heatmap_df["Total Influence"] = heatmap_df[COMPONENTS].abs().sum(axis=1)
    (heatmap_df[["Total Influence"]]
     .sort_values("Total Influence", ascending=False)
     .to_csv(OUT_DIR / "parameter_influence_ranking.csv"))

    # Aggregated CSV
    aggregated_df = pd.concat(aggregated_frames, ignore_index=True)
    aggregated_df.to_csv(OUT_DIR / "period_data_aggregated.csv", index=False)

    print(f"  Processed {len(aggregated_frames)} parameter folder(s).")
    print("  ✅ parameter_period_sensitivity_summary.csv")
    print("  ✅ period_sensitivity_heatmap.png")
    print("  ✅ parameter_influence_ranking.csv")
    print("  ✅ period_data_aggregated.csv")
    return aggregated_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Period deltas vs base value
# ══════════════════════════════════════════════════════════════════════════════
def step2_calculate_deltas(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print(f"[ Step 2 | {CONDITION_LABEL} ] Calculating period deltas vs base ...")
    print("="*60)

    param_col = _find_col(aggregated_df, ["Parameter", "parameter", "Parameter_Name"])
    value_col = _find_col(aggregated_df, ["Value", "Parameter Value", "Parameter_Value",
                                          "param", "parameter"])
    if param_col is None or value_col is None:
        raise ValueError(f"Missing required columns. Found: {list(aggregated_df.columns)}")

    delta_df = (aggregated_df.copy()
                .sort_values([param_col, value_col])
                .reset_index(drop=True))

    for comp in COMPONENTS:
        if comp not in delta_df.columns:
            delta_df[f"{comp}_delta"] = np.nan
            continue

        def _delta(row: pd.Series, _comp=comp) -> float:
            pname = row[param_col]
            base_val = BASE_PARAMS.get(pname)
            if base_val is None:
                return float("nan")
            mask = (delta_df[param_col] == pname) & (
                np.abs(delta_df[value_col] - base_val) < 1e-6)
            if not mask.any():
                return float("nan")
            base_period = delta_df.loc[mask, _comp].values[0]
            return float("nan") if pd.isna(base_period) else row[_comp] - base_period

        delta_df[f"{comp}_delta"] = delta_df.apply(_delta, axis=1)

    delta_df["FoldChange"] = delta_df.apply(
        lambda row: (row[value_col] / BASE_PARAMS[row[param_col]])
        if row[param_col] in BASE_PARAMS else np.nan,
        axis=1,
    )

    delta_df.to_csv(OUT_DIR / "parameter_period_deltas.csv", index=False)
    print("  ✅ parameter_period_deltas.csv")
    return delta_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Plots and summary CSVs
# ══════════════════════════════════════════════════════════════════════════════
def step3_plot_and_summarise(delta_df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print(f"[ Step 3 | {CONDITION_LABEL} ] Plotting fold-change curves & slopes ...")
    print("="*60)

    _ensure_out("parameter_fold_change_plots")

    param_col = _find_col(delta_df, ["Parameter", "parameter", "Parameter_Name"])
    value_col = _find_col(delta_df, ["Value", "Parameter Value", "Parameter_Value",
                                     "param", "parameter"])
    if param_col is None or value_col is None:
        raise ValueError(f"Missing required columns. Found: {list(delta_df.columns)}")

    delta_cols = [f"{c}_delta" for c in COMPONENTS if f"{c}_delta" in delta_df.columns]
    df = delta_df.dropna(subset=delta_cols, how="all").copy() if delta_cols else delta_df.copy()
    parameters = df[param_col].unique()

    # ── Slopes ────────────────────────────────────────────────────────────
    slope_dict: dict[str, dict[str, float | None]] = {}
    for param in parameters:
        pdata = df[df[param_col] == param]
        entry: dict[str, float | None] = {}
        for comp in COMPONENTS:
            if comp not in pdata.columns:
                entry[comp] = None
                continue
            sub = pdata[["FoldChange", comp]].dropna()
            entry[comp] = float(linregress(sub["FoldChange"], sub[comp])[0]) if len(sub) > 1 else None
        slope_dict[param] = entry

    print("  Slope summary (period vs fold-change):")
    for param, s in slope_dict.items():
        print(f"    {param}: { {k: f'{v:.3f}' if v is not None else 'N/A' for k, v in s.items()} }")

    # ── CV ────────────────────────────────────────────────────────────────
    cv_dict: dict[str, dict[str, float | None]] = {}
    for param in parameters:
        pdata = df[df[param_col] == param]
        entry: dict[str, float | None] = {}
        for comp in COMPONENTS:
            if comp not in pdata.columns:
                entry[comp] = None
                continue
            vals = pdata[comp].dropna()
            entry[comp] = float(vals.std() / vals.mean()) if (len(vals) > 1 and vals.mean() != 0) else None
        cv_dict[param] = entry

    # ── Mean slopes + categories ──────────────────────────────────────────
    mean_slopes: dict[str, float] = {}
    for param in parameters:
        valid = [v for v in slope_dict[param].values() if v is not None]
        if valid:
            mean_slopes[param] = sum(valid) / len(valid)

    param_categories: dict[str, str] = {}
    for param, ms in mean_slopes.items():
        if abs(ms) > 0.5:
            param_categories[param] = "Highly Sensitive"
        elif abs(ms) > 0.2:
            param_categories[param] = "Moderately Sensitive"
        else:
            param_categories[param] = "Less Sensitive"

    # ── 2×2 Period vs FoldChange ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    for idx, (ax, comp) in enumerate(zip(axes.flatten(), COMPONENTS)):
        for i, param in enumerate(parameters):
            sub = df[df[param_col] == param]
            if comp in sub.columns and sub[comp].notna().any():
                cv_val = cv_dict[param][comp]
                cv_str = f"  CV={cv_val:.2f}" if cv_val is not None else ""
                ax.plot(sub["FoldChange"], sub[comp],
                        label=f"{param}{cv_str}", marker="o", linewidth=1.4,
                        color=OKABE_ITO[i % len(OKABE_ITO)])
        ax.set_title(f"[{CONDITION_LABEL}] {comp} — Period vs Fold Change")
        ax.set_xlabel("Fold Change")
        ax.set_ylabel("Period (h)")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), ncol=6, fontsize="small")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_DIR / "combined_parameter_period_plot.png", dpi=300)
    plt.close(fig)
    print("  ✅ combined_parameter_period_plot.png")

    # ── Stacked 4-panel slope bar chart ───────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    for idx, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        param_list, slope_list = [], []
        for param in parameters:
            s = slope_dict[param].get(comp)
            if s is not None:
                param_list.append(param)
                slope_list.append(s)
        bar_colors = [SLOPE_COLOR_MAP.get(param_categories.get(p, "Less Sensitive"), "#999999")
                      for p in param_list]
        bars = ax.bar(param_list, slope_list, color=bar_colors)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, rotation=90)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"Slope ({comp})", fontsize=10)
        ax.set_title(f"[{CONDITION_LABEL}] Sensitivity of {comp} Period to Parameter Changes",
                     fontsize=12)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Parameter", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "stacked_component_slope_plots.png", dpi=300)
    plt.close(fig)
    print("  ✅ stacked_component_slope_plots.png")

    # ── Mean slope bar chart ──────────────────────────────────────────────
    sorted_params = sorted(mean_slopes, key=mean_slopes.get, reverse=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(sorted_params,
           [mean_slopes[p] for p in sorted_params],
           color=[SLOPE_COLOR_MAP.get(param_categories.get(p, "Less Sensitive"), "#999999")
                  for p in sorted_params])
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(sorted_params)))
    ax.set_xticklabels(sorted_params, rotation=90, fontsize=8)
    ax.set_ylabel("Mean Slope (Period vs Fold Change)")
    ax.set_title(f"[{CONDITION_LABEL}] Average Sensitivity of Period to Parameter Changes")
    ax.legend(handles=[Patch(facecolor=c, label=lbl) for lbl, c in SLOPE_COLOR_MAP.items()],
              loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "combined_mean_slope_bar.png", dpi=300)
    plt.close(fig)
    print("  ✅ combined_mean_slope_bar.png")

    # ── Summary CSV ───────────────────────────────────────────────────────
    slope_records = [
        {"Parameter": p, "Component": c, "Slope": s}
        for p, comp_slopes in slope_dict.items()
        for c, s in comp_slopes.items()
    ]
    cv_records = [
        {"Parameter": p, "Component": c, "CV": v}
        for p, comp_cvs in cv_dict.items()
        for c, v in comp_cvs.items()
    ]
    mean_slope_df = pd.DataFrame({
        "Parameter": list(mean_slopes.keys()),
        "MeanSlope": list(mean_slopes.values()),
        "Category": [param_categories[p] for p in mean_slopes],
    })
    (pd.DataFrame(slope_records)
     .merge(mean_slope_df, on="Parameter", how="left")
     .merge(pd.DataFrame(cv_records), on=["Parameter", "Component"], how="left")
     .to_csv(OUT_DIR / "parameter_sensitivity_summary.csv", index=False))
    print("  ✅ parameter_sensitivity_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█"*60)
    print(f"  Period Analysis Pipeline  |  Condition: {CONDITION_LABEL}")
    print(f"  Data dir  : {DATA_DIR}")
    print(f"  Out dir   : {OUT_DIR}")
    print("█"*60)

    aggregated_df = step1_load_and_analyse()
    delta_df      = step2_calculate_deltas(aggregated_df)
    step3_plot_and_summarise(delta_df)

    print("\n" + "█"*60)
    print(f"  ✅  {CONDITION_LABEL} pipeline complete.  Outputs → {OUT_DIR}")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
