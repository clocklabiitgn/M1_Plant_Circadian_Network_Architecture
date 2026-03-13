import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# User inputs
# ----------------------------
csv_file_ll = "mean_metrics_ll.csv"
csv_file_ld = "mean_metrics_ld.csv"
outdir = Path("mean_area_eccentricity_plots")
outdir.mkdir(exist_ok=True)

# Column names in CSV
PARAM_COL = "parameter"
FOLD_COL = "fold_change"
AREA_COL = "mean_area"
ECC_COL = "mean_eccentricity"

# Colors for conditions
LL_COLOR = "#1f77b4"
LD_COLOR = "#ff7f0e"


# ----------------------------
# Load data (no pandas dependency)
# ----------------------------
def load_rows(csv_file):
    rows = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ----------------------------
# Compute slopes per parameter
# ----------------------------
def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def compute_slopes(rows, ycol):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(PARAM_COL, "")].append(row)

    results = []
    for param, sub in grouped.items():
        x = np.array([to_float(r.get(FOLD_COL)) for r in sub], dtype=float)
        y = np.array([to_float(r.get(ycol)) for r in sub], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2 or np.unique(x).size < 2:
            slope = np.nan
        else:
            try:
                slope = np.polyfit(x, y, 1)[0]
            except np.linalg.LinAlgError:
                slope = np.nan
        results.append(
            {
                PARAM_COL: param,
                "slope": slope,
            }
        )
    return results


def merge_slopes(slopes_ll, slopes_ld):
    merged = {}
    for row in slopes_ll:
        merged[row[PARAM_COL]] = {
            "ll": row["slope"],
            "ld": np.nan,
        }
    for row in slopes_ld:
        param = row[PARAM_COL]
        if param not in merged:
            merged[param] = {
                "ll": np.nan,
                "ld": row["slope"],
            }
        else:
            merged[param]["ld"] = row["slope"]
    return merged


def plot_slope_bars_dual(merged, title, outfile, y_label):
    def sort_key(item):
        _, data = item
        vals = [data["ll"], data["ld"]]
        mean_val = np.nanmean(vals)
        if np.isnan(mean_val):
            mean_val = np.inf
        return mean_val

    sorted_items = sorted(merged.items(), key=sort_key)
    params = [p for p, _ in sorted_items]
    ll_vals = [d["ll"] for _, d in sorted_items]
    ld_vals = [d["ld"] for _, d in sorted_items]
    x = np.arange(len(params))
    width = 0.38

    plt.figure(figsize=(10, 3))
    plt.bar(x - width / 2, ll_vals, width=width, color=LL_COLOR, label="ll")
    plt.bar(x + width / 2, ld_vals, width=width, color=LD_COLOR, label="ld")
    plt.ylabel(y_label)
    plt.xlabel("Parameter")
    plt.title(title)
    plt.xticks(x, params, rotation=90, fontsize=6)
    plt.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / outfile, dpi=300)
    plt.close()


def sort_by_delta(merged):
    def sort_key(item):
        _, data = item
        ll = data["ll"]
        ld = data["ld"]
        if np.isfinite(ll) and np.isfinite(ld):
            delta = ld - ll
        else:
            delta = np.inf
        return delta

    return sorted(merged.items(), key=sort_key)


def plot_dot(merged, title, outfile, y_label):
    items = sort_by_delta(merged)
    params = [p for p, _ in items]
    ll_vals = [d["ll"] for _, d in items]
    ld_vals = [d["ld"] for _, d in items]

    x = np.arange(len(params))
    plt.figure(figsize=(10, 3))
    plt.scatter(x, ll_vals, color=LL_COLOR, label="ll", s=18)
    plt.scatter(x, ld_vals, color=LD_COLOR, label="ld", s=18)
    plt.axhline(0, color="#bbbbbb", linewidth=0.6)
    plt.xticks(x, params, rotation=90, fontsize=6)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(frameon=False, ncol=2, loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / outfile, dpi=300)
    plt.close()


def plot_dumbbell(merged, title, outfile, y_label):
    items = sort_by_delta(merged)
    params = [p for p, _ in items]
    ll_vals = [d["ll"] for _, d in items]
    ld_vals = [d["ld"] for _, d in items]

    x = np.arange(len(params))
    plt.figure(figsize=(10, 3))
    for i, (ll, ld) in enumerate(zip(ll_vals, ld_vals)):
        if np.isfinite(ll) and np.isfinite(ld):
            plt.plot([x[i], x[i]], [ll, ld], color="#999999", linewidth=0.8, zorder=1)
    plt.scatter(x, ll_vals, color=LL_COLOR, label="ll", s=20, zorder=2)
    plt.scatter(x, ld_vals, color=LD_COLOR, label="ld", s=20, zorder=2)
    plt.axhline(0, color="#bbbbbb", linewidth=0.6)
    plt.xticks(x, params, rotation=90, fontsize=6)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(frameon=False, ncol=2, loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / outfile, dpi=300)
    plt.close()


def plot_slope_chart(merged, title, outfile, y_label):
    items = sort_by_delta(merged)
    params = [p for p, _ in items]
    ll_vals = [d["ll"] for _, d in items]
    ld_vals = [d["ld"] for _, d in items]

    x_left, x_right = 0, 1
    plt.figure(figsize=(6, 0.18 * len(params) + 2))
    for i, (p, ll, ld) in enumerate(zip(params, ll_vals, ld_vals)):
        if np.isfinite(ll) and np.isfinite(ld):
            plt.plot([x_left, x_right], [ll, ld], color="#999999", linewidth=0.8)
        plt.scatter([x_left], [ll], color=LL_COLOR, s=18)
        plt.scatter([x_right], [ld], color=LD_COLOR, s=18)

    plt.xticks([x_left, x_right], ["ll", "ld"])
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outdir / outfile, dpi=300)
    plt.close()


# ----------------------------
# Plots: Rate (slope) of mean area and mean eccentricity vs fold change
# ----------------------------
rows_ll = load_rows(csv_file_ll)
rows_ld = load_rows(csv_file_ld)

area_slopes_ll = compute_slopes(rows_ll, AREA_COL)
area_slopes_ld = compute_slopes(rows_ld, AREA_COL)
area_merged = merge_slopes(area_slopes_ll, area_slopes_ld)
plot_slope_bars_dual(
    area_merged,
    "Rate of Mean Area vs Fold Change",
    "rate_mean_area_vs_fold_change.png",
    "Slope (mean area per fold change)",
)
plot_dot(
    area_merged,
    "Dot Plot: Mean Area vs Fold Change",
    "dot_mean_area_vs_fold_change.png",
    "Slope (mean area per fold change)",
)
plot_dumbbell(
    area_merged,
    "Dumbbell: Mean Area vs Fold Change",
    "dumbbell_mean_area_vs_fold_change.png",
    "Slope (mean area per fold change)",
)
plot_slope_chart(
    area_merged,
    "Slope Chart: Mean Area vs Fold Change",
    "slopechart_mean_area_vs_fold_change.png",
    "Slope (mean area per fold change)",
)

ecc_slopes_ll = compute_slopes(rows_ll, ECC_COL)
ecc_slopes_ld = compute_slopes(rows_ld, ECC_COL)
ecc_merged = merge_slopes(ecc_slopes_ll, ecc_slopes_ld)
plot_slope_bars_dual(
    ecc_merged,
    "Rate of Mean Eccentricity vs Fold Change",
    "rate_mean_eccentricity_vs_fold_change.png",
    "Slope (mean eccentricity per fold change)",
)
plot_dot(
    ecc_merged,
    "Dot Plot: Mean Eccentricity vs Fold Change",
    "dot_mean_eccentricity_vs_fold_change.png",
    "Slope (mean eccentricity per fold change)",
)
plot_dumbbell(
    ecc_merged,
    "Dumbbell: Mean Eccentricity vs Fold Change",
    "dumbbell_mean_eccentricity_vs_fold_change.png",
    "Slope (mean eccentricity per fold change)",
)
plot_slope_chart(
    ecc_merged,
    "Slope Chart: Mean Eccentricity vs Fold Change",
    "slopechart_mean_eccentricity_vs_fold_change.png",
    "Slope (mean eccentricity per fold change)",
)

print("Plots saved in:", outdir)
