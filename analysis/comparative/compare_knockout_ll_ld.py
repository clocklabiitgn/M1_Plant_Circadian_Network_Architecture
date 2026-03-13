#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from openpyxl import Workbook

FONT_SIZE = 20


def _to_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_knockout_csv(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        period_cols = [c for c in fieldnames if c.startswith("Period_")]
        period_info = []
        for c in period_cols:
            suffix = c.split("_", 1)[1]
            try:
                period_num = float(suffix)
            except ValueError:
                period_num = None
            period_info.append((c, period_num))

        def sort_key(item):
            col, num = item
            return (math.inf if num is None else num, col)

        period_info.sort(key=sort_key)
        data = defaultdict(lambda: defaultdict(list))
        for row in reader:
            param = (row.get("Parameter") or "").strip()
            value = row.get("Value")
            value = _to_float(value)
            series = []
            for col, _ in period_info:
                series.append(_to_float(row.get(col)))
            if param:
                data[param][value] = series
        return period_info, data


def sanitize_filename(name):
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "parameter"


def _pick_baseline_value(values, value_zero):
    candidates = [v for v in values if v is not None and v != value_zero]
    if not candidates:
        return None
    return sorted(candidates)[0]


def _delta_series(zero_series, base_series):
    delta = []
    for z, b in zip(zero_series, base_series):
        if z is None or b is None:
            delta.append(None)
        else:
            delta.append(z - b)
    return delta


def build_delta_mean_by_param(data, value_zero, baseline_value=None):
    out = {}
    for param, values in data.items():
        zero_series = values.get(value_zero)
        if zero_series is None:
            continue
        base_value = baseline_value
        if base_value is None:
            base_value = _pick_baseline_value(values.keys(), value_zero)
        if base_value is None:
            continue
        base_series = values.get(base_value)
        if base_series is None:
            continue
        zero_mean = _mean_ignore_none(zero_series)
        base_mean = _mean_ignore_none(base_series)
        if zero_mean is None or base_mean is None:
            continue
        out[param] = base_mean - zero_mean
    return out


def _mean_ignore_none(values):
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _replace_none_with_zero(values):
    cleaned = []
    for v in values:
        cleaned.append(0.0 if v is None else v)
    return cleaned


def _resolve_output_path(out_path, fmt, default_name):
    if out_path.lower().endswith((".png", ".pdf", ".svg")):
        return out_path
    os.makedirs(out_path, exist_ok=True)
    return os.path.join(out_path, f"{default_name}.{fmt}")


def _resolve_output_dir(out_path):
    if out_path.lower().endswith((".png", ".pdf", ".svg")):
        out_dir = os.path.dirname(out_path) or "."
    else:
        out_dir = out_path
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_delta_by_parameter(
    ll_delta_mean,
    ld_delta_mean,
    out_path,
    dpi,
    condition,
):
    params = sorted(set(ll_delta_mean.keys()) | set(ld_delta_mean.keys()))
    if not params:
        raise SystemExit("No parameters with value=0 and baseline found.")

    x = list(range(len(params)))
    x_labels = params

    fig, ax = plt.subplots(figsize=(max(8, len(params) * 0.6), 5))
    bar_width = 0.4 if condition == "both" else 0.6

    if condition in ("ll", "both"):
        y = _replace_none_with_zero([ll_delta_mean.get(p) for p in params])
        offset = -bar_width / 2 if condition == "both" else 0
        ax.bar([xi + offset for xi in x], y, width=bar_width, label="LL mean(Period_*)")
    if condition in ("ld", "both"):
        y = _replace_none_with_zero([ld_delta_mean.get(p) for p in params])
        offset = bar_width / 2 if condition == "both" else 0
        ax.bar([xi + offset for xi in x], y, width=bar_width, label="LD mean(Period_*)")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.set_xlabel("Parameter", fontsize=FONT_SIZE)
    ax.set_ylabel("Mean |Δ Period| per Parameter", fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE, title_fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_dumbbell_delta_mean(
    ll_metric,
    ld_metric,
    out_path,
    dpi,
    sort_by,
    ordered_params=None,
    categories_by_param=None,
):
    params = sorted(set(ll_metric.keys()) | set(ld_metric.keys()))
    rows = []
    ordered = ordered_params if ordered_params is not None else params
    for p in ordered:
        ll_val = ll_metric.get(p)
        ld_val = ld_metric.get(p)
        if ll_val is None or ld_val is None:
            continue
        rows.append((p, ll_val, ld_val))
    if not rows:
        raise SystemExit("No parameters with LL+LD delta means available.")

    if ordered_params is None:
        if sort_by == "ll":
            rows.sort(key=lambda r: r[1])
        elif sort_by == "ld":
            rows.sort(key=lambda r: r[2])
        else:
            rows.sort(key=lambda r: abs(r[1]), reverse=True)
    x = list(range(len(rows)))
    labels = [r[0] for r in rows]
    ll_vals = [r[1] for r in rows]
    ld_vals = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.35), 5))
    for i, (ll_val, ld_val) in enumerate(zip(ll_vals, ld_vals)):
        ax.plot([i, i], [ll_val, ld_val], color="gray", linewidth=1, alpha=0.6, zorder=1)
    ax.scatter(x, ll_vals, color="#0072B2", label="LL", zorder=3)
    ax.scatter(x, ld_vals, color="#E69F00", label="LD", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    if len(rows) > 1:
        ax.set_xlim(-0.45, len(rows) - 0.85)
    else:
        ax.set_xlim(-0.45, 0.45)
    if categories_by_param is not None:
        category_colors = {
            "Class I": "#0072B2",
            "Class II": "#E69F00",
            "Class III": "#999999",
        }
        for tick, p in zip(ax.get_xticklabels(), labels):
            tick.set_color(category_colors[categories_by_param[p]])
    ax.set_xlabel("Parameter", fontsize=FONT_SIZE)
    ax.set_ylabel("Mean |Δ Period| (hours)", fontsize=FONT_SIZE)
    ax.set_title("Knockout Effect of Parameters (LL vs LD)", fontsize=FONT_SIZE)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
    ax.legend(fontsize=FONT_SIZE, title_fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _build_mean_abs_effect(ll_delta_mean, ld_delta_mean):
    params = sorted(set(ll_delta_mean.keys()) | set(ld_delta_mean.keys()))
    ll_abs = {}
    ld_abs = {}
    combined = {}
    for p in params:
        ll_val = ll_delta_mean.get(p)
        ld_val = ld_delta_mean.get(p)
        ll_abs[p] = 0.0 if ll_val is None else abs(ll_val)
        ld_abs[p] = 0.0 if ld_val is None else abs(ld_val)
        vals = []
        if ll_val is not None:
            vals.append(abs(ll_val))
        if ld_val is not None:
            vals.append(abs(ld_val))
        combined[p] = 0.0 if not vals else sum(vals) / len(vals)
    return ll_abs, ld_abs, combined


def _assign_classes(scores, high_threshold, mid_threshold):
    classes = {}
    for p, score in scores.items():
        if score >= high_threshold:
            classes[p] = "Class I"
        elif score >= mid_threshold:
            classes[p] = "Class II"
        else:
            classes[p] = "Class III"
    return classes


def plot_category_effect_bar(
    metric_by_param,
    categories_by_param,
    ordered_params,
    out_path,
    dpi,
    title,
):
    category_colors = {
        "Class I": "#0072B2",    # blue
        "Class II": "#E69F00",   # orange
        "Class III": "#999999",  # gray
    }
    values = [metric_by_param.get(p, 0.0) for p in ordered_params]
    bar_colors = [category_colors[categories_by_param[p]] for p in ordered_params]
    x = list(range(len(ordered_params)))

    fig, ax = plt.subplots(figsize=(max(10, len(ordered_params) * 0.3), 5), dpi=dpi)
    ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_params, rotation=90, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    for tick, p in zip(ax.get_xticklabels(), ordered_params):
        tick.set_color(category_colors[categories_by_param[p]])

    ax.set_xlabel("Parameter", fontsize=FONT_SIZE)
    ax.set_ylabel("Mean |Δ Period| (hours)", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor=category_colors["Class I"], edgecolor="none", label="Class I"),
        Patch(facecolor=category_colors["Class II"], edgecolor="none", label="Class II"),
        Patch(facecolor=category_colors["Class III"], edgecolor="none", label="Class III"),
        
        
    ]
    ax.legend(
        handles=handles,
        title="Category",
        loc="upper right",
        frameon=True,
        fontsize=FONT_SIZE,
        title_fontsize=FONT_SIZE,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def export_delta_mean_excel(
    ll_metric,
    ld_metric,
    out_path,
    class_threshold_high,
    class_threshold_mid,
):
    ll_classes = _assign_classes(ll_metric, class_threshold_high, class_threshold_mid)
    ld_classes = _assign_classes(ld_metric, class_threshold_high, class_threshold_mid)
    _, _, combined_abs = _build_mean_abs_effect(ll_metric, ld_metric)
    combined_classes = _assign_classes(
        combined_abs, class_threshold_high, class_threshold_mid
    )

    wb = Workbook()
    ws_ll = wb.active
    ws_ll.title = "LL"
    ws_ld = wb.create_sheet("LD")

    headers = [
        "Parameter",
        "Mean_Delta_Period",
        "Class_This_Condition",
        "Class_Combined",
    ]
    ws_ll.append(headers)
    ws_ld.append(headers)

    for p in sorted(ll_metric.keys()):
        ws_ll.append([p, ll_metric[p], ll_classes[p], combined_classes.get(p)])
    for p in sorted(ld_metric.keys()):
        ws_ld.append([p, ld_metric[p], ld_classes[p], combined_classes.get(p)])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    wb.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a single delta plot: for each parameter, compute (value=0 - baseline) "
            "for each period and plot across parameters."
        )
    )
    parser.add_argument("--ll", default="knockout_analysis_ll.csv", help="LL CSV path")
    parser.add_argument("--ld", default="knockout_analysis_ld.csv", help="LD CSV path")
    parser.add_argument(
        "--out",
        default="plots_knockout",
        help="Output file path or directory (default: plots_knockout/)",
    )
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--value-zero",
        type=float,
        default=0.0,
        help="Parameter value treated as the knockout (default: 0).",
    )
    parser.add_argument(
        "--baseline-value",
        type=float,
        default=None,
        help="Baseline value to compare against; default picks the smallest non-zero value.",
    )
    parser.add_argument(
        "--condition",
        choices=["ll", "ld", "both"],
        default="both",
        help="Which condition(s) to plot.",
    )
    parser.add_argument(
        "--style",
        choices=["bar", "dumbbell", "manuscript"],
        default="bar",
        help="Plot style: bar (delta mean), dumbbell (delta mean), or manuscript (category bars).",
    )
    parser.add_argument(
        "--sort-by",
        choices=["ll", "ld", "abs-ll"],
        default="ll",
        help="Sort order for dumbbell: ll, ld, or abs-ll.",
    )
    parser.add_argument(
        "--class-threshold-high",
        type=float,
        default=5.0,
        help="Class I threshold for manuscript plots (default: 5.0).",
    )
    parser.add_argument(
        "--class-threshold-mid",
        type=float,
        default=0.5,
        help="Class II threshold for manuscript plots (default: 0.5).",
    )
    parser.add_argument(
        "--excel-out",
        default=None,
        help=(
            "Optional Excel output path (.xlsx). Exports LL and LD mean delta period "
            "results in separate sheets with classes."
        ),
    )
    args = parser.parse_args()

    if args.style == "dumbbell" and args.condition != "both":
        raise SystemExit("--style dumbbell requires --condition both (LL and LD).")

    period_info_ll, ll_data = read_knockout_csv(args.ll)
    period_info_ld, ld_data = read_knockout_csv(args.ld)

    if [c for c, _ in period_info_ll] != [c for c, _ in period_info_ld]:
        raise SystemExit("Period columns differ between LL and LD CSVs.")

    ll_delta_mean = build_delta_mean_by_param(
        ll_data, args.value_zero, args.baseline_value
    )
    ld_delta_mean = build_delta_mean_by_param(
        ld_data, args.value_zero, args.baseline_value
    )
    ll_abs, ld_abs, combined_abs = _build_mean_abs_effect(ll_delta_mean, ld_delta_mean)

    if args.excel_out:
        excel_out = args.excel_out
        if not excel_out.lower().endswith(".xlsx"):
            excel_out = f"{excel_out}.xlsx"
        export_delta_mean_excel(
            ll_abs,
            ld_abs,
            excel_out,
            args.class_threshold_high,
            args.class_threshold_mid,
        )
        print(f"Saved Excel summary to {excel_out}")

    if args.style == "bar":
        out_path = _resolve_output_path(
            args.out, args.format, "knockout_delta_compare"
        )
        plot_delta_by_parameter(
            ll_delta_mean,
            ld_delta_mean,
            out_path,
            args.dpi,
            args.condition,
        )
        print(f"Saved delta plot to {out_path}")
    elif args.style == "dumbbell":
        categories = _assign_classes(
            combined_abs, args.class_threshold_high, args.class_threshold_mid
        )
        ordered_params = sorted(combined_abs.keys(), key=lambda p: combined_abs[p], reverse=True)
        out_path = _resolve_output_path(
            args.out, args.format, "knockout_dumbbell_delta_period"
        )
        plot_dumbbell_delta_mean(
            ll_abs,
            ld_abs,
            out_path,
            args.dpi,
            args.sort_by,
            ordered_params=ordered_params,
            categories_by_param=categories,
        )
        print(f"Saved dumbbell plot to {out_path}")
    else:
        categories = _assign_classes(
            combined_abs, args.class_threshold_high, args.class_threshold_mid
        )
        ordered_params = sorted(combined_abs.keys(), key=lambda p: combined_abs[p], reverse=True)
        out_dir = _resolve_output_dir(args.out)

        combined_out = os.path.join(
            out_dir, f"knockout_effect_combined.{args.format}"
        )
        ll_out = os.path.join(out_dir, f"knockout_effect_ll.{args.format}")
        ld_out = os.path.join(out_dir, f"knockout_effect_ld.{args.format}")

        plot_category_effect_bar(
            combined_abs,
            categories,
            ordered_params,
            combined_out,
            args.dpi,
            "Combined Knockout Effect of Parameters (Mean Absolute Period Change)",
        )
        plot_category_effect_bar(
            ll_abs,
            categories,
            ordered_params,
            ll_out,
            args.dpi,
            "LL Knockout Effect of Parameters (Mean Absolute Period Change)",
        )
        plot_category_effect_bar(
            ld_abs,
            categories,
            ordered_params,
            ld_out,
            args.dpi,
            "LD Knockout Effect of Parameters (Mean Absolute Period Change)",
        )
        print(f"Saved manuscript plots to {out_dir}")


if __name__ == "__main__":
    main()
