#!/usr/bin/env python3
"""
phase_and_expression_plots_per_value.py

Input folder layout:
  parameter_analysis/
    Am7/
      expression_profile_Am7_0.886264.csv
      ...
    K3/
      expression_profile_K3_2.0.csv
      ...

Expected CSV columns (normalized):
  Time,
  CLm_norm, P97m_norm, P51m_norm, EL_norm
(If your files also contain *_raw, that's fine.)

Outputs (created next to this script by default):
  phase_portraits/<param>/phase_<param>_<value>.png + .pdf
  norm_expression_profiles/<param>/exprnorm_<param>_<value>.png + .pdf

Run:
  python phase_and_expression_plots_per_value.py
  python phase_and_expression_plots_per_value.py --root /path/to/parameter_analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Publication-ready styling
# ----------------------------
def set_nature_like_style() -> None:
    """
    Nature-ish defaults: clean axes, inward ticks, consistent line widths.
    Uses a safe default font (DejaVu Sans) to avoid Arial-not-found issues on Linux/HPC.
    """
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "savefig.bbox": "tight",
    })


# ----------------------------
# Helpers
# ----------------------------
VALUE_REGEX = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_param_and_value(fp: Path) -> Tuple[Optional[str], Optional[float]]:
    """
    Tries to parse: expression_profile_<param>_<value>.csv
    """
    name = fp.stem
    # common pattern
    m = re.match(r"expression_profile_(.+?)_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", name)
    if m:
        return m.group(1), float(m.group(2))

    # fallback: last number in filename = value, param in between
    nums = VALUE_REGEX.findall(name)
    if not nums:
        return None, None

    try:
        v = float(nums[-1])
    except ValueError:
        return None, None

    # try to pull param from "expression_profile_<param>_..."
    if name.startswith("expression_profile_"):
        rest = name[len("expression_profile_"):]
        # remove trailing "_<value>"
        rest = re.sub(rf"[_-]{re.escape(nums[-1])}$", "", rest)
        if rest:
            return rest, v

    return None, v


def load_norm_df(fp: Path) -> pd.DataFrame:
    """
    Load expression profile and return standardized columns:
      Time, CLm, P97m, P51m, EL  (from *_norm columns)
    """
    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]

    if "Time" not in df.columns:
        df.insert(0, "Time", np.arange(len(df), dtype=float))

    needed = {
        "CLm": "CLm_norm",
        "P97m": "P97m_norm",
        "P51m": "P51m_norm",
        "EL": "EL_norm",
    }
    for k, col in needed.items():
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {fp.name}. Found: {list(df.columns)}")

    out = df[["Time", "CLm_norm", "P97m_norm", "P51m_norm", "EL_norm"]].copy()
    out = out.rename(columns={
        "CLm_norm": "CLm",
        "P97m_norm": "P97m",
        "P51m_norm": "P51m",
        "EL_norm": "EL",
    })

    # numeric clean
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    if out.empty:
        raise ValueError(f"No valid numeric data after cleaning: {fp.name}")

    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def format_value_for_name(v: float) -> str:
    """
    Stable filename-safe formatting.
    """
    # Avoid scientific notation in filenames when possible.
    if abs(v) >= 1e-3 and abs(v) < 1e4:
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return f"{v:.3e}".replace("+", "")


# ----------------------------
# Plotters
# ----------------------------
def plot_norm_expression(df: pd.DataFrame,
                         param: str,
                         value: float,
                         out_png: Path,
                         out_pdf: Path) -> None:
    comps = ["CLm", "P97m", "P51m", "EL"]

    # Colorblind-friendly palette (Okabe–Ito-like). Chosen explicitly for print clarity.
    comp_colors: Dict[str, str] = {
        "CLm": "#0072B2",   # blue
        "P97m": "#D55E00",  # vermillion
        "P51m": "#009E73",  # green
        "EL":  "#CC79A7",   # purple
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    axes = axes.ravel()

    for ax, comp in zip(axes, comps):
        ax.plot(df["Time"].values, df[comp].values, linewidth=1.8, color=comp_colors[comp])
        ax.set_title(comp, pad=6)
        ax.set_ylabel("Normalized expression")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # keep the y-range consistent for normalized curves
        ax.set_ylim(-0.05, 1.05)

    for ax in axes[2:]:
        ax.set_xlabel("Time (h)")

    fig.suptitle(f"{param} = {value:g} | Normalized expression profiles", y=1.02)

    fig.savefig(out_png, dpi=600)
    # fig.savefig(out_pdf, dpi=600)
    plt.close(fig)


def plot_phase_portraits(df: pd.DataFrame,
                         param: str,
                         value: float,
                         out_png: Path,
                         out_pdf: Path,
                         drop_transient_frac: float = 0.0) -> None:
    """
    6 pairwise phase portraits among 4 components.
    Uses a single high-contrast, colorblind-safe line color; time ordering is shown by line continuity.
    """
    pairs = [
        ("CLm", "P97m"),
        ("CLm", "P51m"),
        ("CLm", "EL"),
        ("P97m", "P51m"),
        ("P97m", "EL"),
        ("P51m", "EL"),
    ]

    if drop_transient_frac > 0:
        n = len(df)
        start = int(np.floor(n * drop_transient_frac))
        dfp = df.iloc[start:].copy()
    else:
        dfp = df

    # A single, print-safe, colorblind-friendly dark blue
    line_color = "#1B3A57"

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    axes = axes.ravel()

    for ax, (xk, yk) in zip(axes, pairs):
        ax.plot(dfp[xk].values, dfp[yk].values, linewidth=1.4, color=line_color)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # normalized signals -> same scale
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(f"{param} = {value:g} | Phase portraits (normalized)", y=1.02)

    fig.savefig(out_png, dpi=600)
    # fig.savefig(out_pdf, dpi=600)
    plt.close(fig)


# ----------------------------
# Main driver
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Per-value normalized expression + phase portrait plotting.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing parameter folders (default: <script_dir>/parameter_analysis).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="expression_profile_*.csv",
        help="CSV filename pattern inside each parameter folder.",
    )
    parser.add_argument(
        "--drop_transient_frac",
        type=float,
        default=0.0,
        help="Drop first fraction of samples for phase plots (e.g., 0.3 drops first 30%).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = (script_dir / "parameter_analysis") if args.root is None else Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    # Output roots (next to script)
    out_phase_root = script_dir / "phase_portraits"
    out_expr_root = script_dir / "norm_expression_profiles"
    ensure_dir(out_phase_root)
    ensure_dir(out_expr_root)

    set_nature_like_style()

    param_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not param_dirs:
        raise RuntimeError(f"No parameter folders found under: {root}")

    total_files = 0
    plotted = 0
    failed = 0

    for param_dir in param_dirs:
        param = param_dir.name

        csv_files = sorted(param_dir.rglob(args.glob))
        if not csv_files:
            continue

        # Create output subfolders per parameter
        phase_param_dir = out_phase_root / param
        expr_param_dir = out_expr_root / param
        ensure_dir(phase_param_dir)
        ensure_dir(expr_param_dir)

        for fp in csv_files:
            total_files += 1
            parsed_param, value = parse_param_and_value(fp)
            # If parsed_param exists and differs, trust folder name as parameter label
            if value is None:
                failed += 1
                continue

            try:
                df = load_norm_df(fp)
                vname = format_value_for_name(value)

                phase_png = phase_param_dir / f"phase_{param}_{vname}.png"
                phase_pdf = phase_param_dir / f"phase_{param}_{vname}.pdf"
                expr_png = expr_param_dir / f"exprnorm_{param}_{vname}.png"
                expr_pdf = expr_param_dir / f"exprnorm_{param}_{vname}.pdf"

                plot_phase_portraits(
                    df=df, param=param, value=value,
                    out_png=phase_png, out_pdf=phase_pdf,
                    drop_transient_frac=args.drop_transient_frac,
                )
                plot_norm_expression(
                    df=df, param=param, value=value,
                    out_png=expr_png, out_pdf=expr_pdf,
                )

                plotted += 1
                print(f"✅ {param}: value={value:g} -> saved phase+expr")
            except Exception as e:
                failed += 1
                print(f"❌ Failed: {fp} | {type(e).__name__}: {e}")

    print("\n--- Summary ---")
    print(f"Root: {root}")
    print(f"Total CSV matched: {total_files}")
    print(f"Plotted successfully: {plotted}")
    print(f"Failed: {failed}")
    print(f"Phase outputs: {out_phase_root}")
    print(f"Expr outputs:  {out_expr_root}")


if __name__ == "__main__":
    main()
