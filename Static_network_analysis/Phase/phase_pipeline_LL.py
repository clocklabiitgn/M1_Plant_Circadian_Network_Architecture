#!/usr/bin/env python3
"""
phase_pipeline_LL.py

COMBINED PIPELINE — LL (Constant Light) condition
==================================================
Step 1: Generate per-value normalized expression profiles and phase portraits
         from parameter_analysis/ CSV data.
Step 2: Analyze the generated phase portrait PNGs for convex-hull area and
         eccentricity changes vs the base parameter value.

Input folder layout:
  parameter_analysis/
    Am7/
      expression_profile_Am7_0.886264.csv
      ...
    K3/
      expression_profile_K3_2.0.csv
      ...

Expected CSV columns (normalized):
  Time, CLm_norm, P97m_norm, P51m_norm, EL_norm

Outputs (all next to this script under  LL_outputs/):
  LL_outputs/phase_portraits/<param>/phase_<param>_<value>.png
  LL_outputs/norm_expression_profiles/<param>/exprnorm_<param>_<value>.png
  LL_outputs/area_ecc_analysis_<date>/
      subplot_metrics.csv
      mean_metrics.csv
      analysis_overview.json
      plots/<param>_area_change.png
      plots/<param>_eccentricity_change.png

LL-specific behaviour vs LD version:
  • CONDITION_LABEL = "LL"  → stamped on all plot titles
  • OUTPUT_ROOT_NAME = "LL_outputs"
  • BASE_PARAMS reflects constant-light parameter values:
      - k1D (dark degradation rate) set to 0         (no dark cycle)
      - d2D, d3D, d5D, d6D (dark degradation rates) set to 0
      - d2L, d3L, d5L, d6L (light degradation rates) retained
      - p1L (light-dependent p1 contribution) retained
      - All other parameters identical to LD baseline

Run:
  python phase_pipeline_LL.py
  python phase_pipeline_LL.py --root /path/to/parameter_analysis --drop_transient_frac 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Base parameter values for LL condition
# Dark-cycle rates zeroed; light-active rates retained.
# ──────────────────────────────────────────────────────────────────────────────
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

CONDITION_LABEL = "LL"
OUTPUT_ROOT_NAME = "LL_outputs"

# ──────────────────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────────────────
PAIRS = [
    ("LHYm", "P97m"),
    ("LHYm", "P51m"),
    ("LHYm", "ELm"),
    ("P97m", "P51m"),
    ("P97m", "ELm"),
    ("P51m", "ELm"),
]

VALUE_REGEX = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def set_nature_like_style() -> None:
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


def parse_param_and_value(fp: Path) -> Tuple[Optional[str], Optional[float]]:
    name = fp.stem
    m = re.match(r"expression_profile_(.+?)_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", name)
    if m:
        return m.group(1), float(m.group(2))
    nums = VALUE_REGEX.findall(name)
    if not nums:
        return None, None
    try:
        v = float(nums[-1])
    except ValueError:
        return None, None
    if name.startswith("expression_profile_"):
        rest = name[len("expression_profile_"):]
        rest = re.sub(rf"[_-]{re.escape(nums[-1])}$", "", rest)
        if rest:
            return rest, v
    return None, v


def load_norm_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]
    if "Time" not in df.columns:
        df.insert(0, "Time", np.arange(len(df), dtype=float))
    needed = {"LHYm": "LHYm_norm", "P97m": "P97m_norm", "P51m": "P51m_norm", "ELm": "ELm_norm"}
    for col in needed.values():
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {fp.name}. Found: {list(df.columns)}")
    out = df[["Time", "LHYm_norm", "P97m_norm", "P51m_norm", "ELm_norm"]].copy()
    out = out.rename(columns={"LHYm_norm": "LHYm", "P97m_norm": "P97m", "P51m_norm": "P51m", "ELm_norm": "ELm"})
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    if out.empty:
        raise ValueError(f"No valid numeric data after cleaning: {fp.name}")
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def format_value_for_name(v: float) -> str:
    if abs(v) >= 1e-3 and abs(v) < 1e4:
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return f"{v:.3e}".replace("+", "")


def plot_norm_expression(df: pd.DataFrame, param: str, value: float,
                         out_png: Path, condition: str) -> None:
    comps = ["LHYm", "P97m", "P51m", "ELm"]
    # Slightly warmer palette for LL to visually distinguish from LD
    comp_colors: Dict[str, str] = {
        "LHYm": "#E69F00",   # orange
        "P97m": "#56B4E9",   # sky-blue
        "P51m": "#009E73",   # green
        "ELm":   "#CC79A7",   # purple
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    axes = axes.ravel()
    for ax, comp in zip(axes, comps):
        ax.plot(df["Time"].values, df[comp].values, linewidth=1.8, color=comp_colors[comp])
        ax.set_title(comp, pad=6)
        ax.set_ylabel("Normalized expression")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-0.05, 1.05)
    for ax in axes[2:]:
        ax.set_xlabel("Time (h)")
    fig.suptitle(f"[{condition}] {param} = {value:g} | Normalized expression profiles", y=1.02)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def plot_phase_portraits(df: pd.DataFrame, param: str, value: float,
                         out_png: Path, condition: str,
                         drop_transient_frac: float = 0.0) -> None:
    if drop_transient_frac > 0:
        start = int(np.floor(len(df) * drop_transient_frac))
        dfp = df.iloc[start:].copy()
    else:
        dfp = df
    # Slightly warmer line color for LL
    line_color = "#7B2D00"
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    axes = axes.ravel()
    for ax, (xk, yk) in zip(axes, PAIRS):
        ax.plot(dfp[xk].values, dfp[yk].values, linewidth=1.4, color=line_color)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f"[{condition}] {param} = {value:g} | Phase portraits (normalized)", y=1.02)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def run_plotting(root: Path, out_root: Path, glob_pat: str, drop_transient_frac: float) -> Path:
    out_phase_root = out_root / "phase_portraits"
    out_expr_root = out_root / "norm_expression_profiles"
    ensure_dir(out_phase_root)
    ensure_dir(out_expr_root)

    set_nature_like_style()

    param_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not param_dirs:
        raise RuntimeError(f"No parameter folders found under: {root}")

    total_files = plotted = failed = 0
    for param_dir in param_dirs:
        param = param_dir.name
        csv_files = sorted(param_dir.rglob(glob_pat))
        if not csv_files:
            continue
        phase_param_dir = out_phase_root / param
        expr_param_dir = out_expr_root / param
        ensure_dir(phase_param_dir)
        ensure_dir(expr_param_dir)
        for fp in csv_files:
            total_files += 1
            _, value = parse_param_and_value(fp)
            if value is None:
                failed += 1
                continue
            try:
                df = load_norm_df(fp)
                vname = format_value_for_name(value)
                phase_png = phase_param_dir / f"phase_{param}_{vname}.png"
                expr_png = expr_param_dir / f"exprnorm_{param}_{vname}.png"
                plot_phase_portraits(df=df, param=param, value=value,
                                     out_png=phase_png, condition=CONDITION_LABEL,
                                     drop_transient_frac=drop_transient_frac)
                plot_norm_expression(df=df, param=param, value=value,
                                     out_png=expr_png, condition=CONDITION_LABEL)
                plotted += 1
                print(f"  ✅ {param}: value={value:g}")
            except Exception as e:
                failed += 1
                print(f"  ❌ Failed: {fp} | {type(e).__name__}: {e}")

    print(f"\n[Step 1 summary] Total CSV: {total_files}  Plotted: {plotted}  Failed: {failed}")
    return out_phase_root


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PHASE PORTRAIT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

class PhasePortraitAnalyzer:
    def __init__(self, base_dir: str, output_dir: str, base_params: dict):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.base_params = base_params
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.parameter_folders: list[str] = []
        self.parameter_frames: dict[str, list] = {}

    @staticmethod
    def _extract_param_value(filename: str, param_name: str) -> Optional[float]:
        base = os.path.splitext(os.path.basename(filename))[0]
        pattern = rf"^phase_{re.escape(param_name)}_(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$"
        match = re.match(pattern, base)
        if match:
            return float(match.group("value"))
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", base)
        if tokens:
            return float(tokens[-1])
        return None

    def load_parameter_folders(self):
        print("Loading parameter folders...")
        out_base = os.path.basename(self.output_dir)
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if (os.path.isdir(item_path)
                    and item != out_base
                    and not item.startswith("area_ecc_analysis_")):
                self.parameter_folders.append(item)
        print(f"  Found {len(self.parameter_folders)} parameter folders")

    def load_parameter_frames(self):
        print("Loading parameter frames...")
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        for param_folder in self.parameter_folders:
            folder_path = os.path.join(self.base_dir, param_folder)
            param_frames = []
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in valid_ext:
                    v = self._extract_param_value(fname, param_folder)
                    if v is not None:
                        param_frames.append({"path": fpath, "value": v})
            param_frames.sort(key=lambda x: x["value"])
            self.parameter_frames[param_folder] = param_frames
            print(f"  {param_folder}: {len(param_frames)} frames")

    @staticmethod
    def _split_subplots(image: np.ndarray, rows: int = 2, cols: int = 3,
                        border_frac: float = 0.08) -> list[np.ndarray]:
        h, w = image.shape[:2]
        tile_h, tile_w = h // rows, w // cols
        tiles = []
        for r in range(rows):
            for c in range(cols):
                tile = image[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w].copy()
                cx = int(tile_w * border_frac)
                cy = int(tile_h * border_frac)
                tile = tile[cy:tile.shape[0]-cy, cx:tile.shape[1]-cx]
                tiles.append(tile)
        return tiles

    @staticmethod
    def _line_mask(tile: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) if tile.ndim == 3 else tile
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    @staticmethod
    def _area_and_eccentricity(mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            return np.nan, np.nan
        points = np.column_stack([xs, ys]).astype(np.int32)
        hull = cv2.convexHull(points)
        area = float(cv2.contourArea(hull))
        coords = points.astype(float)
        centered = coords - coords.mean(axis=0)
        eigvals = np.sort(np.linalg.eigvalsh(np.cov(centered, rowvar=False)))[::-1]
        ecc = float(np.sqrt(max(0.0, 1.0 - (eigvals[1] / eigvals[0])))) if eigvals[0] > 0 else np.nan
        return area, ecc

    def analyze_frame(self, frame_path: str) -> list[dict]:
        frame = np.array(Image.open(frame_path))
        tiles = self._split_subplots(frame)
        metrics = []
        for tile in tiles:
            mask = self._line_mask(tile)
            area, ecc = self._area_and_eccentricity(mask)
            metrics.append({"area": area, "eccentricity": ecc})
        return metrics

    def analyze_parameter_folder(self, param_name: str) -> Optional[dict]:
        print(f"  Analyzing: {param_name} ...")
        frames = self.parameter_frames.get(param_name, [])
        if not frames:
            return None

        base_value = self.base_params.get(param_name)
        base_index = None
        if base_value is not None:
            values = np.array([f["value"] for f in frames], dtype=float)
            base_index = int(np.argmin(np.abs(values - base_value)))
            base_value = values[base_index]

        per_frame = [{"param_value": f["value"], "metrics": self.analyze_frame(f["path"])} for f in frames]
        base_metrics = per_frame[base_index]["metrics"] if base_index is not None else None

        rows_subplots, rows_mean = [], []
        for entry in per_frame:
            value = entry["param_value"]
            fold_change = (value / base_value) if (base_value and base_value != 0) else np.nan
            deltas_area, deltas_ecc = [], []

            for idx, metric in enumerate(entry["metrics"]):
                pair_label = f"{PAIRS[idx][0]} vs {PAIRS[idx][1]}"
                area, ecc = metric["area"], metric["eccentricity"]
                delta_area = (area - base_metrics[idx]["area"]) if base_metrics else np.nan
                delta_ecc = (ecc - base_metrics[idx]["eccentricity"]) if base_metrics else np.nan
                deltas_area.append(delta_area)
                deltas_ecc.append(delta_ecc)
                rows_subplots.append({
                    "condition": CONDITION_LABEL,
                    "parameter": param_name, "param_value": value,
                    "fold_change": fold_change, "subplot_index": idx + 1,
                    "pair": pair_label, "area": area, "eccentricity": ecc,
                    "delta_area": delta_area, "delta_eccentricity": delta_ecc,
                    "base_param_value": base_value,
                })

            rows_mean.append({
                "condition": CONDITION_LABEL,
                "parameter": param_name, "param_value": value,
                "fold_change": fold_change,
                "mean_area": float(np.nanmean([m["area"] for m in entry["metrics"]])),
                "mean_eccentricity": float(np.nanmean([m["eccentricity"] for m in entry["metrics"]])),
                "mean_delta_area": float(np.nanmean(deltas_area)),
                "mean_delta_eccentricity": float(np.nanmean(deltas_ecc)),
                "base_param_value": base_value,
            })

        return {"parameter": param_name, "base_value": base_value,
                "subplot_rows": rows_subplots, "mean_rows": rows_mean}

    def analyze_all_parameters(self):
        if not self.parameter_folders:
            self.load_parameter_folders()
        if not self.parameter_frames:
            self.load_parameter_frames()

        all_subplots, all_means, all_results = [], [], []
        for param_name in self.parameter_folders:
            result = self.analyze_parameter_folder(param_name)
            if not result:
                continue
            all_subplots.extend(result["subplot_rows"])
            all_means.extend(result["mean_rows"])
            all_results.append(result)

        if not all_subplots:
            print("No analysis results.")
            return

        df_subplots = pd.DataFrame(all_subplots)
        df_means = pd.DataFrame(all_means)
        df_subplots.to_csv(os.path.join(self.output_dir, "subplot_metrics.csv"), index=False)
        df_means.to_csv(os.path.join(self.output_dir, "mean_metrics.csv"), index=False)
        with open(os.path.join(self.output_dir, "analysis_overview.json"), "w") as f:
            json.dump({"condition": CONDITION_LABEL,
                       "parameters": [r["parameter"] for r in all_results]}, f, indent=2)
        self._plot_results(df_subplots, df_means)
        print(f"\n[Step 2] Analysis complete → {self.output_dir}")

    def _plot_results(self, df_subplots: pd.DataFrame, df_means: pd.DataFrame):
        for param_name in sorted(df_subplots["parameter"].unique()):
            df_p = df_subplots[df_subplots["parameter"] == param_name]
            df_m = df_means[df_means["parameter"] == param_name]
            self._plot_change(df_p, df_m, param_name, "delta_area",
                              "Change in area (px²)", f"{param_name}_area_change.png")
            self._plot_change(df_p, df_m, param_name, "delta_eccentricity",
                              "Change in eccentricity", f"{param_name}_eccentricity_change.png")

    def _plot_change(self, df_param, df_mean, param_name, metric, ylabel, filename):
        plt.figure(figsize=(8, 5))
        for idx, pair in enumerate(PAIRS, start=1):
            df_sub = df_param[df_param["subplot_index"] == idx].sort_values("fold_change")
            plt.plot(df_sub["fold_change"], df_sub[metric], marker="o", linewidth=1.2,
                     label=f"{pair[0]} vs {pair[1]}")
        df_m_sorted = df_mean.sort_values("fold_change")
        plt.plot(df_m_sorted["fold_change"], df_m_sorted[f"mean_{metric}"],
                 color="black", linewidth=2.0, marker="o", label="Mean")
        plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        plt.xlabel("Fold change vs base", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.title(f"[{CONDITION_LABEL}] {param_name}: {ylabel}", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(loc="best", fontsize=14)
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Combined phase portrait pipeline — {CONDITION_LABEL} condition."
    )
    parser.add_argument("--root", type=str, default=None,
                        help="Root directory containing parameter folders (default: <script_dir>/parameter_analysis).")
    parser.add_argument("--glob", type=str, default="expression_profile_*.csv",
                        help="CSV filename pattern inside each parameter folder.")
    parser.add_argument("--drop_transient_frac", type=float, default=0.0,
                        help="Drop first fraction for phase plots (e.g. 0.3 = first 30%%).")
    parser.add_argument("--skip_plotting", action="store_true",
                        help="Skip Step 1 (use existing phase_portraits folder).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = (script_dir / "parameter_analysis") if args.root is None else Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    out_root = script_dir / OUTPUT_ROOT_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Pipeline condition : {CONDITION_LABEL}")
    print(f"  Input root         : {root}")
    print(f"  Output root        : {out_root}")
    print(f"{'='*60}\n")

    # ── Step 1 ─────────────────────────────────────────────────────────────
    if not args.skip_plotting:
        print("[ Step 1 ] Generating expression + phase portrait plots ...")
        phase_portraits_dir = run_plotting(root, out_root, args.glob, args.drop_transient_frac)
    else:
        phase_portraits_dir = out_root / "phase_portraits"
        if not phase_portraits_dir.exists():
            raise FileNotFoundError(f"--skip_plotting set but folder not found: {phase_portraits_dir}")
        print(f"[ Step 1 ] Skipped. Using existing: {phase_portraits_dir}")

    # ── Step 2 ─────────────────────────────────────────────────────────────
    print("\n[ Step 2 ] Analyzing phase portraits for area / eccentricity ...")
    date_str = datetime.now().strftime("%Y-%m-%d")
    analysis_out = str(out_root / f"area_ecc_analysis_{date_str}")

    analyzer = PhasePortraitAnalyzer(
        base_dir=str(phase_portraits_dir),
        output_dir=analysis_out,
        base_params=BASE_PARAMS,
    )
    analyzer.load_parameter_folders()
    analyzer.load_parameter_frames()
    analyzer.analyze_all_parameters()

    print(f"\n{'='*60}")
    print(f"  ✅  {CONDITION_LABEL} pipeline complete.")
    print(f"  Outputs → {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
