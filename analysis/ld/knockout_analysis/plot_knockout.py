from __future__ import annotations

from pathlib import Path
import math
import re
import pandas as pd
import matplotlib.pyplot as plt

# Function to define model parameters
def define_parameters():
    params = {
        "v1": 4.8318, "q1a": 1.4266, "q3a": 8.9432, "q4a": 5.9277, 
        "K1": 0.1943, "K2": 1.6138, "k1L": 0.2866, "k1D": 0.213, "p1": 0.8672, "p1L": 0.2378,
        "d1": 0.7843, "q1b": 3.575, "q3b": 5.5899, "q4b": 8.954, "v2": 1.6822, "K3": 2.2275,
        "K4": 0.40, "K5": 0.37, "k2": 0.35, "p2": 0.7858, "d2D": 0.3712, "d2L": 0.2917,
        "v3": 1.113, "K6": 0.4944, "K7": 2.4087, "k3": 0.5819, "p3": 0.6142, "d3D": 0.5026,
        "d3L": 0.5431, "v4": 2.5012, "K8": 0.3262, "K9": 1.7974, "K10": 1.1889, "k4": 0.925,
        "p4": 1.126, "de1": 0.0022, "de2": 0.4741, "de3": 0.3765, "de4": 0.398, "de5": 0.0003,
        "Ap3": 0.3868, "Am7": 0.5503, "Ak7": 1.125, "q2": 0.5767, "kmpac": 137, "kd": 7,
        "v5": 0.1129, "K11": 0.3322, "k5": 0.1591, "p5": 0.5293, "d5D": 0.4404, "d5L": 5.0712,
        "g1": 0.001, "g2": 0.18, "K12": 0.86, "Bp4": 0.4147, "Bm8": 0.7728, "Bk8": 0.1732,
        "kmpbc": 7162, "Cp5": 0.4567, "Cm9": 0.867, "Ck9": 0.3237, "kmcc": 13406, "K13": 8.04866,
        "K14": 0.347569, "v6": 0.438785, "k6": 0.328256, "K15": 0.336558, "p6": 0.264064,
        "d6D": 0.576502, "d6L": 0.190108}
    return params

# ---------- Your publication style ----------
def set_pub_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "savefig.bbox": "tight",
    })


# ---------- Helper: extract param name from "<param>_baseline.csv" ----------
def infer_param_name(fp: Path) -> str:
    # Works even if param contains underscores.
    name = fp.name
    if name.endswith("_baseline.csv"):
        return name[:-len("_baseline.csv")]
    if name.endswith("_knockout.csv"):
        return name[:-len("_knockout.csv")]
    return fp.stem


def plot_norm_profiles_for_param(
    param_name: str,
    baseline_df: pd.DataFrame,
    knockout_df: pd.DataFrame,
    outdir: Path,
    base_value: float | None = None,
    mode: str = "knockout_vs_base",
) -> None:
    # Components you saved in the CSVs
    components = ["CLm", "P97m", "P51m", "EL"]

    # --- color scheme: cividis (consistent across parameters) ---
    cmap = plt.get_cmap("cividis")
    # Choose two well-separated points in cividis
    colors = {
        "baseline": cmap(0.15),
        "knockout": cmap(0.85),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    axes = axes.ravel()

    if base_value is None:
        base_label = "base"
    else:
        base_label = f"base ({param_name}={base_value:g})"

    for ax, comp in zip(axes, components):
        x_b = baseline_df["Time"].to_numpy()
        y_b = baseline_df[f"{comp}_norm"].to_numpy()

        x_k = knockout_df["Time"].to_numpy()
        y_k = knockout_df[f"{comp}_norm"].to_numpy()

        ax.plot(x_b, y_b, lw=1.3, color=colors["baseline"], label=base_label)
        ax.plot(x_k, y_k, lw=1.3, color=colors["knockout"], label=f"knockout ({param_name}=0)")

        ax.set_title(comp)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized expression")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # One legend for the whole figure (top-center), like your phase-portrait style
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(f"Normalized expression: {param_name} (base vs knockout)", y=1.08)

    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{param_name}_norm_base_vs_knockout.pdf"
    png = outdir / f"{param_name}_norm_base_vs_knockout.png"
    fig.savefig(pdf, dpi=600)
    fig.savefig(png, dpi=600)
    plt.close(fig)


def plot_raw_profiles_for_param(
    param_name: str,
    baseline_df: pd.DataFrame,
    knockout_df: pd.DataFrame,
    outdir: Path,
    base_value: float | None = None,
    mode: str = "knockout_vs_base",
) -> None:
    components = ["CLm", "P97m", "P51m", "EL"]

    cmap = plt.get_cmap("cividis")
    colors = {
        "baseline": cmap(0.15),
        "knockout": cmap(0.85),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    axes = axes.ravel()

    if base_value is None:
        base_label = "base"
    else:
        base_label = f"base ({param_name}={base_value:g})"

    for ax, comp in zip(axes, components):
        x_b = baseline_df["Time"].to_numpy()
        y_b = baseline_df[f"{comp}_raw"].to_numpy()

        x_k = knockout_df["Time"].to_numpy()
        y_k = knockout_df[f"{comp}_raw"].to_numpy()

        ax.plot(x_b, y_b, lw=1.3, color=colors["baseline"], label=base_label)
        ax.plot(x_k, y_k, lw=1.3, color=colors["knockout"], label=f"knockout ({param_name}=0)")

        ax.set_title(comp)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Raw expression")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(f"Raw expression: {param_name} (base vs knockout)", y=1.08)

    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{param_name}_raw_base_vs_knockout.pdf"
    png = outdir / f"{param_name}_raw_base_vs_knockout.png"
    fig.savefig(pdf, dpi=600)
    fig.savefig(png, dpi=600)
    plt.close(fig)


def main() -> None:
    set_pub_style()
    params = define_parameters()

    root = Path("knockout_results")
    profiles_dir = root / "expression_profiles"
    outdir_norm = root / "plots_pub_norm"
    outdir_raw = root / "plots_pub_raw"

    if not profiles_dir.exists():
        raise FileNotFoundError(f"Cannot find: {profiles_dir.resolve()}\n"
                                f"Expected CSVs like <param>_baseline.csv and <param>_knockout.csv.")

    baseline_files = sorted(profiles_dir.glob("*_baseline.csv"))

    if not baseline_files:
        raise FileNotFoundError(f"No '*_baseline.csv' files found in {profiles_dir.resolve()}")

    for base_fp in baseline_files:
        param = infer_param_name(base_fp)
        ko_fp = profiles_dir / f"{param}_knockout.csv"
        if not ko_fp.exists():
            print(f"[skip] Missing knockout file for {param}: {ko_fp.name}")
            continue

        base_df = pd.read_csv(base_fp)
        ko_df = pd.read_csv(ko_fp)

        # Basic sanity checks
        required_cols = {
            "Time",
            "CLm_norm",
            "P97m_norm",
            "P51m_norm",
            "EL_norm",
            "CLm_raw",
            "P97m_raw",
            "P51m_raw",
            "EL_raw",
        }
        if not required_cols.issubset(set(base_df.columns)) or not required_cols.issubset(set(ko_df.columns)):
            print(f"[skip] Columns missing for {param}. Found columns baseline={list(base_df.columns)}")
            continue

        base_value = params.get(param)
        plot_norm_profiles_for_param(param, base_df, ko_df, outdir_norm, base_value=base_value)
        plot_raw_profiles_for_param(param, base_df, ko_df, outdir_raw, base_value=base_value)

    print(f"Done. Saved plots to: {outdir_norm.resolve()}")
    print(f"Done. Saved plots to: {outdir_raw.resolve()}")


if __name__ == "__main__":
    main()
