#!/usr/bin/env python3
"""analyse_and _plot.py

Publication-ready comparative plotting for parameter-sweep runs.

Folder layout (auto-detected by default):
- This script is in some directory.
- In the SAME directory, you have one or more roots matching parameter_analysis_*/, e.g.
    parameter_analysis_ld/
    parameter_analysis_ll/
- Inside each root, you have one folder per parameter (folder name == parameter key), e.g.
    parameter_analysis_ld/q4a/
- Each parameter folder contains expression CSVs like:
    expr_<param>_<value>.csv
  Example:
    expr_Ak7_1.125000.csv

Expected CSV columns (CLm may also appear as LHYm; ELm may also appear as EL):
  Time,
  CLm_raw (or LHYm_raw), CLm_norm (or LHYm_norm),
  P97m_raw, P97m_norm,
  P51m_raw, P51m_norm,
  ELm_raw (or EL_raw),  ELm_norm (or EL_norm)

Plot annotations always use: CLm, P97m, P51m, ELm

Outputs per parameter folder:
  combined_plots/
    combined_expression_<param>_<mode>.pdf/png
    combined_phase_<param>_<mode>.pdf/png
    chosen_values_<param>.txt

Run-level output (next to script):
  parameter_analysis_run_<timestamp>.log

Usage:
  # default root = <script_dir>/parameter_analysis
  python "analyse_and _plot.py" --n_values 5 --mode raw

  # explicit root
  python "analyse_and _plot.py" --root "/path/to/parameter_analysis" --n_values 4 --mode norm

  # explicit values (mapped to nearest available values)
  python "analyse_and _plot.py" --values "1.0,2.0,5.0" --mode raw
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Base parameter values (DO NOT EDIT)
# ---------------------------
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



# ---------------------------
# Publication-friendly styling
# ---------------------------
def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "savefig.bbox": "tight",
    })


# ---------------------------
# Logging
# ---------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    """Configure a file + console logger."""
    logger = logging.getLogger("param_analysis")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers (helpful if run in notebooks/imported)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------
# Parsing helpers
# ---------------------------
VALUE_REGEX = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_value_from_filename(fp: Path, param_name: str) -> Optional[float]:
    """Extract numeric parameter value from filename."""
    stem = fp.stem

    m = re.search(rf"{re.escape(param_name)}\s*=\s*{VALUE_REGEX.pattern}", stem)
    if m:
        return float(m.group(1))

    m = re.search(rf"{re.escape(param_name)}[_-]\s*{VALUE_REGEX.pattern}", stem)
    if m:
        return float(m.group(1))

    nums = VALUE_REGEX.findall(stem)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None

    return None


def pick_component_columns(df: pd.DataFrame, mode: str) -> Dict[str, str]:
    suffix = "_raw" if mode == "raw" else "_norm"

    # Each entry: (canonical_key, [candidates in priority order])
    component_aliases = [
        ("CLm",  [f"CLm{suffix}",  f"LHYm{suffix}"]),
        ("P97m", [f"P97m{suffix}"]),
        ("P51m", [f"P51m{suffix}"]),
        ("ELm",  [f"ELm{suffix}",  f"EL{suffix}"]),
    ]

    colmap: Dict[str, str] = {}
    for key, candidates in component_aliases:
        for c in candidates:
            if c in df.columns:
                colmap[key] = c
                break
        if key not in colmap:
            raise ValueError(
                f"Missing column for '{key}' (tried {candidates}). Found: {list(df.columns)}"
            )

    return colmap


def load_expression_file(fp: Path, mode: str) -> pd.DataFrame:
    """Load expression file and standardize columns to: Time, CLm, P97m, P51m, EL"""
    df: Optional[pd.DataFrame] = None
    for sep in [",", "\t", None]:
        try:
            trial = pd.read_csv(fp, sep=sep, engine="python")
            if trial.shape[1] >= 5:
                df = trial
                break
        except Exception:
            continue

    if df is None or df.empty:
        raise ValueError(f"Could not read file: {fp}")

    df.columns = [str(c).strip() for c in df.columns]

    if "Time" not in df.columns:
        df.insert(0, "Time", np.arange(len(df), dtype=float))

    colmap = pick_component_columns(df, mode)

    out = df[["Time", colmap["CLm"], colmap["P97m"], colmap["P51m"], colmap["ELm"]]].copy()
    out = out.rename(columns={
        colmap["CLm"]: "CLm",
        colmap["P97m"]: "P97m",
        colmap["P51m"]: "P51m",
        colmap["ELm"]: "ELm",
    }) # type: ignore

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna()
    if out.empty:
        raise ValueError(f"No valid numeric rows after cleaning: {fp}")

    return out


# ---------------------------
# Selection helpers
# ---------------------------
def nearest_value(values: List[float], target: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr[np.argmin(np.abs(arr - target))])


def choose_values_by_fold_targets(
    available_values: List[float],
    base_value: float,
    n_values: int
) -> List[float]:
    """
    Choose symmetric values around base:
    lower (base/f), base, higher (base*f)
    """
    values = sorted(available_values)
    if not values:
        return []

    # Define symmetric folds
    if n_values <= 3:
        folds = [0.5, 1.0, 2.0]
    elif n_values == 4:
        folds = [0.2, 0.5, 2.0, 5.0]
    else:
        folds = [0.2, 0.5, 1.0, 2.0, 5.0]

    folds = folds[:n_values]

    targets = [base_value * f for f in folds]

    chosen = [nearest_value(values, t) for t in targets]

    # Remove duplicates while preserving order
    uniq = []
    for v in chosen:
        if v not in uniq:
            uniq.append(v)

    return uniq


def choose_values_by_user_targets(available_values: List[float], requested_values: List[float]) -> List[float]:
    """Map requested values to nearest available values (unique, stable order)."""
    values = sorted(available_values)
    if not values:
        return []

    chosen = [nearest_value(values, v) for v in requested_values]
    uniq: List[float] = []
    for v in chosen:
        if v not in uniq:
            uniq.append(v)
    return uniq


# ---------------------------
# Output helpers
# ---------------------------
def ensure_outdir(param_dir: Path) -> Path:
    outdir = param_dir / "combined_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ---------------------------
# Plotters
# ---------------------------
def plot_combined_expression(param_name: str,
                             series: Dict[float, pd.DataFrame],
                             outdir: Path,
                             mode: str,
                             show_last_hours: Optional[float] = None) -> None:
    fig, axes = plt.subplots(
        2, 2,
        figsize=(2.35, 2.25),
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': 0.75, 'wspace': 0.24}
    )
    axes = axes.ravel()

    comps = ["CLm", "P97m", "P51m", "ELm"]

    vals = sorted(series.keys())
    cmap = plt.get_cmap("viridis")  # type: ignore # colorblind-safe
    colors = {v: cmap(i / max(1, len(vals) - 1)) for i, v in enumerate(vals)}

    ylab = "Expression (a.u.)" if mode == "raw" else "Normalized expression"

    for ax, comp in zip(axes, comps):
        for v in vals:
            df = series[v]
            dff = df
            if show_last_hours is not None:
                tmax = float(df["Time"].max())
                dff = df[df["Time"] >= (tmax - show_last_hours)]

            ax.plot(
                dff["Time"].values,
                dff[comp].values,
                linewidth=1.6,
                color=colors[v],
                label=f"{param_name}={v:g}",
            )

        ax.set_title(comp, pad=6)
        ax.set_ylabel("")
        if comp in ("P97m", "ELm"):
            ax.tick_params(labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[2:]:
        ax.set_xlabel("Time (h)")

    plt.subplots_adjust(top=0.70, left=0.21, right=0.98, bottom=0.11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(vals)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
        columnspacing=0.9,
        handlelength=2.0,
        borderaxespad=0.2,
    )

    fig.suptitle(
        f"Expression profiles across {param_name} ({mode})",
        y=0.985,
        fontsize=11,
    )

    fig.supylabel(ylab, x=0.015, y=0.44, fontsize=10)

    pdf = outdir / f"combined_expression_{param_name}_{mode}.pdf"
    png = outdir / f"combined_expression_{param_name}_{mode}.png"
    fig.savefig(pdf, dpi=600)
    fig.savefig(png, dpi=600)
    plt.close(fig)


def plot_combined_phase_portraits(param_name: str,
                                  series: Dict[float, pd.DataFrame],
                                  outdir: Path,
                                  mode: str,
                                  drop_transient_frac: float = 0.0) -> None:
    pairs = [
        ("CLm", "P97m"),
        ("CLm", "P51m"),
        ("CLm", "ELm"),
        ("P97m", "P51m"),
        ("P97m", "ELm"),
        ("P51m", "ELm"),
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(3.55, 2.85),
        sharex=False,
        sharey=False,
        gridspec_kw={"hspace": 0.72, "wspace": 0.36},
    )
    axes = axes.ravel()

    vals = sorted(series.keys())
    cmap = plt.get_cmap("cividis")
    colors = {v: cmap(i / max(1, len(vals) - 1)) for i, v in enumerate(vals)}

    for i, (ax, (xk, yk)) in enumerate(zip(axes, pairs)):
        for v in vals:
            df = series[v]
            if drop_transient_frac > 0:
                n = len(df)
                start = int(math.floor(n * drop_transient_frac))
                dfp = df.iloc[start:]
            else:
                dfp = df

            ax.plot(
                dfp[xk].values,
                dfp[yk].values,
                linewidth=1.4,
                color=colors[v],
                label=f"{param_name}={v:g}",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

        row, _ = divmod(i, 3)
        if row == 0:
            ax.set_xlabel(xk, labelpad=-2)
        else:
            ax.set_xlabel(xk, labelpad=1)
        ax.set_ylabel(yk, labelpad=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(vals)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.88),
        columnspacing=0.9,
        handlelength=2.0,
        borderaxespad=0.2,
    )

    fig.suptitle(
        f"Phase portraits across {param_name} ({mode})",
        y=0.965,
        fontsize=11,
    )

    plt.subplots_adjust(top=0.76, left=0.12, right=0.99, bottom=0.16)
    pdf = outdir / f"combined_phase_{param_name}_{mode}.pdf"
    png = outdir / f"combined_phase_{param_name}_{mode}.png"
    fig.savefig(pdf, dpi=600)
    fig.savefig(png, dpi=600)
    plt.close(fig)


# ---------------------------
# Per-folder processing
# ---------------------------
def process_param_folder(param_dir: Path,
                         n_values: int,
                         user_values: Optional[List[float]],
                         file_glob: str,
                         mode: str,
                         show_last_hours: Optional[float],
                         drop_transient_frac: float,
                         logger: logging.Logger) -> None:
    param_name = param_dir.name
    logger.info(f"--- Parameter: {param_name} | Folder: {param_dir} ---")

    if param_name not in BASE_PARAMS:
        logger.warning(f"Folder '{param_name}' not found in BASE_PARAMS. Skipping.")
        return

    files = sorted(param_dir.glob(file_glob))
    logger.info(f"Matched {len(files)} files with glob '{file_glob}'")
    if not files:
        logger.warning("No files matched. Skipping.")
        return

    value_to_file: Dict[float, Path] = {}
    skipped = 0
    for fp in files:
        v = parse_value_from_filename(fp, param_name)
        if v is None:
            skipped += 1
            continue
        value_to_file[v] = fp

    if skipped:
        logger.info(f"Skipped {skipped} files (could not parse value from filename)")

    if not value_to_file:
        logger.warning("No parseable values found in filenames. Skipping.")
        return

    available_values = sorted(value_to_file.keys())
    logger.info(f"Parsed {len(available_values)} unique values")

    base_value = float(BASE_PARAMS[param_name])

    if user_values:
        chosen = choose_values_by_user_targets(available_values, user_values)
        logger.info(f"Base value: {base_value}")
        logger.info(f"Requested values: {user_values}")
        logger.info(f"Chosen nearest available: {chosen}")
    else:
        chosen = choose_values_by_fold_targets(available_values, base_value, n_values)
        logger.info(f"Base value: {base_value}")
        logger.info(f"Chosen (fold targets, nearest available): {chosen}")

    series: Dict[float, pd.DataFrame] = {}
    for v in chosen:
        fp = value_to_file.get(v)
        if fp is None:
            continue
        try:
            series[v] = load_expression_file(fp, mode=mode)
            logger.info(f"Loaded: {fp.name} | value={v}")
        except Exception as e:
            logger.warning(f"Failed loading {fp.name} | value={v} | error={e}")

    logger.info(f"Loaded {len(series)} series successfully")
    if len(series) < 2:
        logger.warning("Not enough valid series to compare (need >=2). Skipping plots.")
        return

    outdir = ensure_outdir(param_dir)

    # Save reproducibility note
    try:
        with open(outdir / f"chosen_values_{param_name}.txt", "w", encoding="utf-8") as f:
            f.write(f"Parameter: {param_name}\n")
            f.write(f"Base value: {base_value}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Glob: {file_glob}\n")
            f.write(f"show_last_hours: {show_last_hours}\n")
            f.write(f"drop_transient_frac: {drop_transient_frac}\n")
            f.write("Chosen values (as found in filenames):\n")
            for vv in sorted(series.keys()):
                f.write(f"{vv}\n")
            f.write("\nFiles used:\n")
            for vv in sorted(series.keys()):
                f.write(f"{value_to_file[vv].name}\n")
    except Exception as e:
        logger.warning(f"Could not write chosen_values file: {e}")

    plot_combined_expression(param_name, series, outdir, mode, show_last_hours=show_last_hours)
    plot_combined_phase_portraits(param_name, series, outdir, mode, drop_transient_frac=drop_transient_frac)
    logger.info(f"Saved plots to: {outdir}")


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine expression + phase portrait plots across selected parameter values per parameter folder."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=("Root directory containing parameter folders. If not provided, defaults to "
              "'<script_dir>/parameter_analysis'."),
    )
    parser.add_argument("--n_values", type=int, default=4,
                        help="How many values to compare per parameter (default: 4).")
    parser.add_argument("--values", type=str, default=None,
                        help="Comma-separated explicit values to use (e.g., '1.0,2.0,5.0').")
    parser.add_argument("--glob", type=str, default="expr_*.csv",
                        help="File pattern inside each parameter folder (default: expr_*.csv).")
    parser.add_argument("--mode", type=str, choices=["raw", "norm"], default="raw",
                        help="Which columns to plot: *_raw or *_norm (default: raw).")
    parser.add_argument("--show_last_hours", type=float, default=None,
                        help="If provided, plot only last X hours in expression plots.")
    parser.add_argument("--drop_transient_frac", type=float, default=0.0,
                        help="Drop first fraction of samples for phase portraits (e.g., 0.3 drops first 30%%).")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.root is not None:
        roots = [Path(args.root).expanduser().resolve()]
    else:
        roots = sorted(script_dir.glob("parameter_analysis_*"))
        roots = [r for r in roots if r.is_dir()]
        if not roots:
            # fallback to legacy name
            fallback = script_dir / "parameter_analysis"
            if fallback.exists():
                roots = [fallback]

    if not roots:
        print("[ERROR] No parameter_analysis_* folders found and no --root provided.")
        sys.exit(1)

    for root in roots:
        if not root.exists():
            print(f"[ERROR] Root does not exist: {root}")
            sys.exit(1)

    user_values: Optional[List[float]] = None
    if args.values:
        try:
            user_values = [float(x.strip()) for x in args.values.split(",") if x.strip()]
        except ValueError:
            print("[ERROR] --values must be a comma-separated list of numbers.")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = script_dir / f"parameter_analysis_run_{timestamp}.log"
    logger = setup_logger(log_path)

    logger.info("=== Parameter analysis run started ===")
    logger.info(f"Script: {Path(__file__).resolve()}")
    logger.info(f"Roots: {[str(r) for r in roots]}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Glob: {args.glob}")
    logger.info(f"n_values: {args.n_values}")
    logger.info(f"Explicit values: {user_values if user_values else 'None'}")
    logger.info(f"show_last_hours: {args.show_last_hours}")
    logger.info(f"drop_transient_frac: {args.drop_transient_frac}")
    logger.info(f"Log file: {log_path}")

    set_pub_style()

    for root in roots:
        logger.info(f"=== Processing root: {root} ===")
        param_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        logger.info(f"Found {len(param_dirs)} parameter folders under {root.name}")

        for param_dir in param_dirs:
            process_param_folder(
                param_dir=param_dir,
                n_values=args.n_values,
                user_values=user_values,
                file_glob=args.glob,
                mode=args.mode,
                show_last_hours=args.show_last_hours,
                drop_transient_frac=args.drop_transient_frac,
                logger=logger,
            )

    logger.info("=== Parameter analysis run finished ===")


if __name__ == "__main__":
    main()