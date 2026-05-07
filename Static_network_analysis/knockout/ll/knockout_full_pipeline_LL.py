"""
knockout_full_pipeline_LL.py
════════════════════════════
Full sequential knockout analysis pipeline — CONSTANT LIGHT (LL) condition.

Tasks performed in order:
  1. Summary bar plots  — per-component ΔPeriod subplots + combined mean |ΔPeriod| bar chart
  2. Expression profiles — normalised & raw waveform overlays (base vs knockout) per parameter

Input files expected (edit paths in CONFIG section below if needed):
  • knockout_analysis_ll.csv          — knockout period data for LL
  • knockout_results/expression_profiles/<param>_baseline.csv
  • knockout_results/expression_profiles/<param>_knockout.csv

All outputs are written under:  knockout_output_LL/
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these paths if your files live elsewhere
# ─────────────────────────────────────────────────────────────────────────────
CONDITION          = "LL"
KNOCKOUT_CSV       = "knockout_results_ll/knockout_analysis_ll.csv"   # period-summary CSV
PROFILES_DIR       = Path("knockout_results_ll/expression_profiles")
OUTDIR             = Path("knockout_output_LL")

# Categorisation thresholds (mean |ΔPeriod| in hours)
THRESHOLD_LOW      = 0.05    # below → Class III  |  above → Class II
# Class I = any component loses rhythm (period = 0 or NaN after knockout)

# Minimum visible bar height for Class-I parameters whose |ΔPeriod| rounds to 0
MIN_BAR_HEIGHT     = 0.5

DPI_SUMMARY        = 600     # dpi for bar-chart outputs
DPI_PROFILES       = 600     # dpi for per-parameter profile outputs

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PARAMETERS  (updated M1 set — used for profile-plot axis labels)
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = {
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

# Parameters that are purely fixed/normalisation constants and must be
# excluded from knockout scoring (they are never set to 0 biologically).
FIXED_PARAMS = {"eta1", "eta2"}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────
def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family":       ["Arial", "DejaVu Sans"],
        "font.size":         13,
        "axes.labelsize":    14,
        "axes.titlesize":    16,
        "legend.fontsize":   13,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "axes.linewidth":    1,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "savefig.bbox":      "tight",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })

set_pub_style()
OUTDIR.mkdir(parents=True, exist_ok=True)

# Category colours (shared by all plots in this script)
COLOR_MAP = {"Class I": "tab:red", "Class II": "tab:orange", "Class III": "tab:gray"}

# Profile plot colours (cividis, two well-separated points)
_cmap = plt.get_cmap("cividis")
PROFILE_COLORS = {"baseline": _cmap(0.15), "knockout": _cmap(0.85)}

PERIOD_COLS = ["Period_LHYm", "Period_P97m", "Period_P51m", "Period_ELm"]
PROFILE_COMPONENTS = ["LHYm", "P97m", "P51m", "ELm"]

# Maps internal column-based names → publication annotation labels
DISPLAY_NAMES = {
    "LHYm":        "CLm",
    "Period_LHYm": "CLm",
    "P97m":        "P97m",
    "Period_P97m": "P97m",
    "P51m":        "P51m",
    "Period_P51m": "P51m",
    "ELm":         "ELm",
    "Period_ELm":  "ELm",
}

print(f"{'='*60}")
print(f"  Knockout analysis pipeline — {CONDITION} condition")
print(f"  Output directory: {OUTDIR.resolve()}")
print(f"{'='*60}\n")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — SUMMARY BAR PLOTS
# ═════════════════════════════════════════════════════════════════════════════
print("── Task 1: Summary bar plots ──────────────────────────────")

# ── Load & validate CSV ──────────────────────────────────────────────────────
df = pd.read_csv(KNOCKOUT_CSV)
for col in ["Parameter", "Value", "Label"] + PERIOD_COLS:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in {KNOCKOUT_CSV}")

# ── Compute ΔPeriod per parameter ────────────────────────────────────────────
grouped            = df.groupby("Parameter")
summary_data       = []
core_sensitive_flags = {}

for param, group in grouped:
    if param in FIXED_PARAMS:
        continue
    originals = group.loc[group["Label"] == "baseline"]
    knockouts = group.loc[group["Label"] == "knockout"]
    if len(originals) != 1 or len(knockouts) != 1:
        continue
    original = originals.iloc[0]
    knockout = knockouts.iloc[0]
    changes = [
        (knockout[c] - original[c]) if pd.notna(knockout[c]) else -original[c]
        for c in PERIOD_COLS
    ]
    summary_data.append([param] + changes)
    core_sensitive_flags[param] = any(
        pd.isna(knockout[c]) or knockout[c] == 0 for c in PERIOD_COLS
    )

summary_df = (
    pd.DataFrame(summary_data, columns=["Parameter"] + PERIOD_COLS)
    .set_index("Parameter")
)
mean_abs_change = summary_df.abs().mean(axis=1).sort_values(ascending=False)

# ── Categorise parameters ────────────────────────────────────────────────────
def categorize(param: str) -> str:
    if core_sensitive_flags.get(param, False):
        return "Class I"
    elif mean_abs_change[param] > THRESHOLD_LOW:
        return "Class II"
    return "Class III"

categories = mean_abs_change.index.to_series().apply(categorize)

# ── Plot 1a: four-panel per-component subplots ───────────────────────────────
fig1, axes = plt.subplots(4, 1, figsize=(7.8, 9.6), sharex=True)
for i, comp in enumerate(PERIOD_COLS):
    sorted_series = summary_df[comp].reindex(
        summary_df[comp].abs().sort_values(ascending=False).index
    )
    colors = [COLOR_MAP[categories[idx]] for idx in sorted_series.index]
    sorted_series.plot(kind="bar", ax=axes[i], color=colors)
    axes[i].set_title(f"Period change in {DISPLAY_NAMES.get(comp, comp)} ({CONDITION})")
    axes[i].set_ylabel("Δ Period (hours)")
    axes[i].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[i].set_xticks(range(len(sorted_series)))
    axes[i].set_xticklabels(sorted_series.index, rotation=90, ha="center", va="top")
    for lbl in axes[i].get_xticklabels():
        lbl.set_color(COLOR_MAP[categories[lbl.get_text()]])
    axes[i].grid(axis="x", linestyle=":", color="lightgray", linewidth=0.5)

fig1.suptitle(
    f"Sorted period change — top influential knockouts ({CONDITION})", y=0.995
)
plt.tight_layout(rect=[0, 0, 1, 0.985])
p1a = OUTDIR / f"knockout_component_subplots_{CONDITION}.png"
fig1.savefig(p1a, dpi=DPI_SUMMARY)
plt.close(fig1)
print(f"  Saved: {p1a}")

# ── Reorder by category group, then build combined bar chart ─────────────────
ordered_params = []
for grp in ["Class I", "Class II", "Class III"]:
    ordered_params.extend(categories[categories == grp].index.tolist())

mean_abs_change = mean_abs_change.reindex(ordered_params)
categories      = categories.reindex(ordered_params)
bar_colors      = [COLOR_MAP[categories[p]] for p in mean_abs_change.index]

# Minimum visible height for Class-I zero bars
plot_values = mean_abs_change.copy()
for param in plot_values.index:
    if categories[param] == "Class I" and abs(plot_values[param]) < 1e-6:
        plot_values[param] = MIN_BAR_HEIGHT

# Save CSV summary
summary_out = OUTDIR / f"knockout_analysis_summary_{CONDITION}.csv"
pd.DataFrame({
    "Parameter":          plot_values.index,
    "Mean_Abs_Change":    mean_abs_change.values,
    "Plotted_Bar_Height": plot_values.values,
    "Category":           categories.values,
}).to_csv(summary_out, index=False)
print(f"  Saved: {summary_out}")

# ── Plot 1b: combined mean |ΔPeriod| bar chart ───────────────────────────────
figsize2 = (min(10.5, max(9.5, len(plot_values) * 0.34)), 3.7)
fig2, ax2 = plt.subplots(figsize=figsize2)
plot_values.plot(kind="bar", ax=ax2, color=bar_colors)
ax2.set_title(
    f"Combined knockout effect — mean |ΔPeriod| ({CONDITION})", pad=10
)
ax2.set_ylabel("Mean |Δ Period| (hours)")
ax2.set_xlabel("Parameter")
ax2.tick_params(axis="x", rotation=90)
for lbl in ax2.get_xticklabels():
    lbl.set_color(COLOR_MAP[categories[lbl.get_text()]])
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[k]) for k in COLOR_MAP]
ax2.legend(handles, COLOR_MAP.keys(), title="Category", frameon=True)
for i, (param, val) in enumerate(mean_abs_change.items()):
    if val <= -20:
        ax2.text(i, val - 0.5, "*", color="black", ha="center", va="top", fontsize=14)
ax2.set_ylim(0, plot_values.max() + 2)
fig2.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.52)
p1b = OUTDIR / f"knockout_combined_effect_{CONDITION}.png"
fig2.savefig(p1b, dpi=DPI_SUMMARY)
plt.close(fig2)
print(f"  Saved: {p1b}")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — PER-PARAMETER EXPRESSION PROFILES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n── Task 2: Per-parameter expression profiles ({CONDITION}) ──")

outdir_norm = OUTDIR / "plots_pub_norm"
outdir_raw  = OUTDIR / "plots_pub_raw"
outdir_norm.mkdir(parents=True, exist_ok=True)
outdir_raw.mkdir(parents=True, exist_ok=True)

# Switch to smaller font for the 2×2 profile panels
plt.rcParams.update({
    "font.size":       10,
    "axes.labelsize":  10,
    "axes.titlesize":  10,
    "legend.fontsize":  8,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

REQUIRED_COLS = {
    "Time",
    *[f"{c}_norm" for c in PROFILE_COMPONENTS],
    *[f"{c}_raw"  for c in PROFILE_COMPONENTS],
}

def infer_param_name(fp: Path) -> str:
    name = fp.name
    for suffix in ("_baseline_ll.csv", "_knockout_ll.csv", "_baseline.csv", "_knockout.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return fp.stem

def _plot_2x2(
    param_name: str,
    base_df: pd.DataFrame,
    ko_df: pd.DataFrame,
    col_suffix: str,    # "norm" or "raw"
    ylabel: str,
    outdir: Path,
    base_value: float | None,
) -> None:
    base_label = (
        f"base ({param_name}={base_value:g})" if base_value is not None else "base"
    )
    fig, axes = plt.subplots(
        2, 2,
        figsize=(3.9, 3.4),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.58, "wspace": 0.22},
    )
    axes = axes.ravel()
    for ax, comp in zip(axes, PROFILE_COMPONENTS):
        col = f"{comp}_{col_suffix}"
        ax.plot(base_df["Time"].to_numpy(), base_df[col].to_numpy(),
                lw=1.3, color=PROFILE_COLORS["baseline"], label=base_label)
        ax.plot(ko_df["Time"].to_numpy(),   ko_df[col].to_numpy(),
                lw=1.3, color=PROFILE_COLORS["knockout"],
                label=f"knockout ({param_name}=0)")
        ax.set_title(DISPLAY_NAMES.get(comp, comp))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[2:]:
        ax.set_xlabel("Time (hrs)")
    for ax in (axes[1], axes[3]):
        ax.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.91),
        columnspacing=1.2, handlelength=2.0, borderaxespad=0.2,
    )
    fig.suptitle(
        f"{ylabel}: {param_name} — base vs knockout ({CONDITION})",
        y=0.985, fontsize=11,
    )
    plt.subplots_adjust(top=0.74, left=0.18, right=0.98, bottom=0.13)
    fig.supylabel(ylabel, x=0.04, y=0.44, fontsize=10)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{param_name}_{col_suffix}_base_vs_knockout"
    fig.savefig(outdir / f"{stem}.pdf", dpi=DPI_PROFILES)
    fig.savefig(outdir / f"{stem}.png", dpi=DPI_PROFILES)
    plt.close(fig)

if not PROFILES_DIR.exists():
    print(f"  [skip] Profiles directory not found: {PROFILES_DIR.resolve()}")
else:
    baseline_files = sorted(PROFILES_DIR.glob("*_baseline_ll.csv")) or \
                     sorted(PROFILES_DIR.glob("*_baseline.csv"))
    if not baseline_files:
        print(f"  [skip] No baseline CSV files found in {PROFILES_DIR.resolve()}")
    else:
        _use_ll = baseline_files[0].name.endswith("_baseline_ll.csv")
        _ko_suffix = "_knockout_ll.csv" if _use_ll else "_knockout.csv"
        n_ok = 0
        for base_fp in baseline_files:
            param = infer_param_name(base_fp)
            ko_fp = PROFILES_DIR / f"{param}{_ko_suffix}"
            if not ko_fp.exists():
                print(f"  [skip] No knockout file for: {param}")
                continue
            base_df = pd.read_csv(base_fp)
            ko_df   = pd.read_csv(ko_fp)
            if not REQUIRED_COLS.issubset(set(base_df.columns)) or \
               not REQUIRED_COLS.issubset(set(ko_df.columns)):
                print(f"  [skip] Missing columns for: {param}")
                continue
            bv = PARAMS.get(param)
            _plot_2x2(param, base_df, ko_df, "norm",
                      "Normalized expression", outdir_norm, bv)
            _plot_2x2(param, base_df, ko_df, "raw",
                      "Raw expression", outdir_raw, bv)
            n_ok += 1
        print(f"  Processed {n_ok} parameter(s).")
        print(f"  Normalised plots → {outdir_norm.resolve()}")
        print(f"  Raw plots        → {outdir_raw.resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Pipeline complete — {CONDITION} condition.")
print(f"  All outputs in: {OUTDIR.resolve()}")
print(f"{'='*60}")
