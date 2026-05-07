# --- Plot period vs parameter value for each parameter and component ---

import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

# --- Shared publication style ---
def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

plot_values_len = 0  # Initialize global for dynamic sizing
set_pub_style()

# Create output directory for plots
plot_dir = "parameter_fold_change_plots"
os.makedirs(plot_dir, exist_ok=True)

# Reload the delta CSVs for LL and LD
import pandas as pd
df_ll = pd.read_csv("parameter_period_deltas_ll.csv")
df_ld = pd.read_csv("parameter_period_deltas_ld.csv")
df_ll["Condition"] = "LL"
df_ld["Condition"] = "LD"
df_all = pd.concat([df_ll, df_ld], ignore_index=True)

# Drop rows with missing original period values (i.e., where delta computation failed)
df_clean = df_all.dropna(
    subset=["CLm_delta", "P97m_delta", "P51m_delta", "ELm_delta"], how="all"
).copy()

components = ['CLm', 'P97m', 'P51m', 'ELm']

# Define base parameter values for fold change calculation
base_values = {
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

# Compute FoldChange column (handle both "Value" and "Parameter Value" inputs)
value_col = "Value" if "Value" in df_clean.columns else "Parameter_Value"
base_series = df_clean["Parameter"].map(base_values)
df_clean["FoldChange"] = df_clean[value_col] / base_series

# --- Calculate slope of period change vs fold change for each parameter and component ---
from scipy.stats import linregress
from matplotlib.lines import Line2D

# Plot period vs parameter value for each parameter and component (LL + LD)
parameters = df_clean["Parameter"].unique()
conditions = ["LL", "LD"]
parameters_all_by_condition = {
    condition: df_all.loc[df_all["Condition"] == condition, "Parameter"].dropna().unique()
    for condition in conditions
}

# Create subplot grid for all components (line plots across parameters)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)  # Temporary fixed size until plot_values_len computed
axes = axes.flatten()
colors = plt.cm.tab20.colors
line_styles = {"LL": "-", "LD": "--"}
markers = {"LL": "o", "LD": "s"}

for idx, comp in enumerate(components):
    ax = axes[idx]
    for i, param in enumerate(parameters):
        for condition in conditions:
            subset = df_clean[
                (df_clean["Parameter"] == param) & (df_clean["Condition"] == condition)
            ]
            if subset[comp].notna().any():
                ax.plot(
                    subset["FoldChange"],
                    subset[comp],
                    color=colors[i % len(colors)],
                    linestyle=line_styles.get(condition, "-"),
                    marker=markers.get(condition, "o"),
                    alpha=0.9,
                )
    ax.set_title(f"{comp} Period vs Parameter Value (LL vs LD)")
    ax.set_xlabel("Fold Change")
    ax.set_ylabel("Period (hours)")
    ax.grid(True)

# Add legends: parameters (colors) and conditions (line styles)
param_handles = [
    Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=param)
    for i, param in enumerate(parameters)
]
cond_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        lw=2,
        linestyle=line_styles[c],
        marker=markers[c],
        label=c,
    )
    for c in conditions
]

legend_params = fig.legend(
    handles=param_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=6,
    fontsize="small",
    title="Parameter",
)
fig.add_artist(legend_params)
fig.legend(
    handles=cond_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(conditions),
    fontsize="small",
    title="Condition",
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("combined_parameter_period_plot.png", dpi=300)
plt.close()

print("✅ Combined period variation plot saved as 'combined_parameter_period_plot.png'")


# Store slope tables for optional comparative plots
slope_tables = {}


def analyze_condition(df_condition, condition_label):
    # Calculate slope of period change vs fold change for each parameter and component
    parameters = parameters_all_by_condition.get(
        condition_label, df_condition["Parameter"].dropna().unique()
    )
    slope_dict = {}
    for param in parameters:
        param_data = df_condition[df_condition["Parameter"] == param]
        slope_entry = {}
        for comp in components:
            subset = param_data[[comp, "FoldChange"]].dropna()
            if len(subset) > 1:
                slope, _, _, _, _ = linregress(subset["FoldChange"], subset[comp])
                slope_entry[comp] = slope
            else:
                slope_entry[comp] = None
        slope_dict[param] = slope_entry

    # Optionally print slope summary
    print(f"Parameter-wise slope of period vs fold change ({condition_label}):")
    for param, comp_slopes in slope_dict.items():
        print(f"{param}: {comp_slopes}")

    # Compute coefficient of variation (CV) for each parameter and component
    cv_dict = {}
    for param in parameters:
        param_data = df_condition[df_condition["Parameter"] == param]
        cv_entry = {}
        for comp in components:
            comp_values = param_data[comp].dropna()
            if len(comp_values) > 1:
                mean_val = comp_values.mean()
                std_val = comp_values.std()
                cv = std_val / mean_val if mean_val != 0 else None
            else:
                cv = None
            cv_entry[comp] = cv
        cv_dict[param] = cv_entry

    # --- Combined slope bar plot: Mean slope across all components ---
    suffix = condition_label.lower()
    combined_slope_plot = f"combined_mean_slope_bar_{suffix}.png"
    mean_slopes = {}
    for param in parameters:
        slopes = [slope_dict.get(param, {}).get(comp) for comp in components]
        valid_slopes = [s for s in slopes if s is not None]
        if valid_slopes:
            mean_slopes[param] = sum(valid_slopes) / len(valid_slopes)

    # Define categories based on slope thresholds (example thresholds)
    param_categories = {}
    for param, mean_slope in mean_slopes.items():
        if abs(mean_slope) > 0.5:
            param_categories[param] = "Highly Sensitive"
        elif abs(mean_slope) > 0.2:
            param_categories[param] = "Moderately Sensitive"
        else:
            param_categories[param] = "Less Sensitive"

# Compute plot_values_len from highly sensitive params (global for figures)
    global plot_values_len
    highly_sensitive_params = [p for p, cat in param_categories.items() if cat == "Highly Sensitive"]
    plot_values_len = len(highly_sensitive_params)
    print(f"Number of highly sensitive parameters (plot_values_len): {plot_values_len}")

    # Save slope values and categories to CSV
    slope_records = []
    for param, comp_slopes in slope_dict.items():
        for comp, slope_val in comp_slopes.items():
            slope_records.append(
                {
                    "Parameter": param,
                    "Component": comp,
                    "Slope": slope_val,
                }
            )
    slope_df = pd.DataFrame(slope_records, columns=["Parameter", "Component", "Slope"])

    mean_slope_df = pd.DataFrame(
        {
            "Parameter": list(mean_slopes.keys()),
            "MeanSlope": list(mean_slopes.values()),
            "Category": [param_categories[p] for p in mean_slopes.keys()],
        }
    )

    # Flatten CV dictionary
    cv_records = []
    for param, comp_cvs in cv_dict.items():
        for comp, cv_val in comp_cvs.items():
            cv_records.append({"Parameter": param, "Component": comp, "CV": cv_val})
    cv_df = pd.DataFrame(cv_records)

    # Merge with slope summary
    summary_df = slope_df.merge(mean_slope_df, on="Parameter", how="left")
    summary_df = summary_df.merge(cv_df, on=["Parameter", "Component"], how="left")

    summary_csv = f"parameter_sensitivity_summary_{suffix}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Slope values and categories saved as '{summary_csv}'")
    slope_tables[condition_label] = slope_df.copy()

    # Replace individual slope plots with stacked subplot
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7.8, 10.5), sharex=True)
    #figsize = (max(7, plot_values_len * 0.35), 5)
    for idx, comp in enumerate(components):
        ax = axes[idx]
        param_list = []
        slope_list = []
        for param in parameters:
            slope = slope_dict.get(param, {}).get(comp)
            if slope is not None:
                param_list.append(param)
                slope_list.append(slope)
        # Color coding by param_categories
        color_map = {
            "Highly Sensitive": "crimson",
            "Moderately Sensitive": "goldenrod",
            "Less Sensitive": "gray",
        }
        bar_colors = [
            color_map.get(param_categories.get(p, "Less Sensitive"), "gray")
            for p in param_list
        ]
        bars = ax.bar(param_list, slope_list, color=bar_colors)
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
        ax.set_ylabel(f"Slope ({comp})", fontsize=10)
        ax.grid(axis="x", which="both", linestyle="--", linewidth=0.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"Sensitivity of {comp} Period to Parameter Changes ({condition_label})",
            fontsize=12,
        )
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)

    axes[-1].set_xlabel("Parameter", fontsize=12)
    plt.tight_layout()
    stacked_plot = f"stacked_component_slope_plots_{suffix}.png"
    plt.savefig(stacked_plot, dpi=300)
    plt.close()
    print(f"✅ Stacked slope bar plot saved as '{stacked_plot}'")

    # Plot combined mean slope bar plot with color by category
    sorted_params = sorted(mean_slopes, key=mean_slopes.get, reverse=True)
    sorted_slopes = [mean_slopes[p] for p in sorted_params]

    color_map = {
        "Highly Sensitive": "crimson",
        "Moderately Sensitive": "goldenrod",
        "Less Sensitive": "gray",
    }
    bar_colors = [color_map[param_categories[p]] for p in sorted_params]

    plot_values = sorted_slopes
    dynamic_width = min(10.5, max(9.5, len(plot_values) * 0.34))
    plt.figure(figsize=(7.8,10.5))
    ax = plt.gca()
    ax.bar(sorted_params, sorted_slopes, color=bar_colors, width=0.85)
    ax.set_xlim(-0.5, len(sorted_params) - 0.5)
    ax.set_xticks(range(len(sorted_params)))
    ax.set_xticklabels(sorted_params, rotation=90, ha="center", va="top")

    for tick, param in zip(ax.get_xticklabels(), sorted_params):
        tick.set_color(color_map[param_categories[param]])

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map["Less Sensitive"], label="Less Sensitive"),
        Patch(facecolor=color_map["Moderately Sensitive"], label="Moderately Sensitive"),
        Patch(facecolor=color_map["Highly Sensitive"], label="Highly Sensitive"),
    ]
    ax.legend(handles=legend_handles, title="Category", loc="upper right", frameon=True)

    plt.ylabel("Mean Slope (Period vs Fold Change)")
    plt.xlabel("Parameter")
    plt.title(f"Average sensitivity of period to parameter changes ({condition_label})", pad=10)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.52)
    plt.savefig(combined_slope_plot, dpi=300)
    plt.close()

    print(f"✅ Combined mean slope bar plot saved as '{combined_slope_plot}'")

    return mean_slopes


mean_slopes_by_condition = {}
for condition in conditions:
    df_condition = df_clean[df_clean["Condition"] == condition]
    mean_slopes_by_condition[condition] = analyze_condition(df_condition, condition)

# Combined mean slope plot (LL vs LD) in a single figure
all_params = sorted(
    {
        p
        for cond_slopes in mean_slopes_by_condition.values()
        for p in cond_slopes.keys()
    }
)
ll_vals = [mean_slopes_by_condition.get("LL", {}).get(p) for p in all_params]
ld_vals = [mean_slopes_by_condition.get("LD", {}).get(p) for p in all_params]

import numpy as np

x = np.arange(len(all_params))
width = 0.38

plt.figure(figsize=(max(7, plot_values_len * 0.35), 5))
plt.bar(x - width / 2, ll_vals, width, label="LL", color="steelblue")
plt.bar(x + width / 2, ld_vals, width, label="LD", color="darkorange")
plt.xticks(x, all_params, rotation=90, fontsize=8)
plt.ylabel("Mean Slope (Period vs Fold Change)")
plt.title("Average Sensitivity of Period to Parameter Changes (LL vs LD)")
plt.legend()
plt.tight_layout()
plt.savefig("combined_mean_slope_bar_ll_ld.png", dpi=300)
plt.close()
print("✅ Combined mean slope bar plot (LL vs LD) saved as 'combined_mean_slope_bar_ll_ld.png'")


# --- Additional comparative plot snippets (publication-friendly) ---

# 1) Period distribution by condition (box + jitter) for each component
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()
for idx, comp in enumerate(components):
    ax = axes[idx]
    data_ll = df_clean[df_clean["Condition"] == "LL"][comp].dropna()
    data_ld = df_clean[df_clean["Condition"] == "LD"][comp].dropna()
    ax.boxplot([data_ll, data_ld], tick_labels=["LL", "LD"], showfliers=False)
    # light jitter points for context
    ax.plot(
        [1] * len(data_ll),
        data_ll,
        "o",
        alpha=0.25,
        markersize=3,
        color="steelblue",
    )
    ax.plot(
        [2] * len(data_ld),
        data_ld,
        "o",
        alpha=0.25,
        markersize=3,
        color="darkorange",
    )
    ax.set_title(f"{comp} Period Distribution (LL vs LD)")
    ax.set_ylabel("Period (hours)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig("period_distribution_boxplots_ll_ld.png", dpi=300)
plt.close()
print("✅ Period distribution boxplots saved as 'period_distribution_boxplots_ll_ld.png'")


# 2) Heatmap of component slopes per parameter (one heatmap per condition)
def plot_slope_heatmap(slope_df, condition_label):
    pivot = slope_df.pivot(index="Parameter", columns="Component", values="Slope")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(f"Slope Heatmap (Period vs Fold Change) - {condition_label}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Slope")
    plt.tight_layout()
    out = f"slope_heatmap_{condition_label.lower()}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Slope heatmap saved as '{out}'")


for condition in conditions:
    if condition in slope_tables:
        plot_slope_heatmap(slope_tables[condition], condition)


# 3) LL vs LD mean slope comparison scatter (per-parameter)
ll_vals = np.array([mean_slopes_by_condition.get("LL", {}).get(p) for p in all_params])
ld_vals = np.array([mean_slopes_by_condition.get("LD", {}).get(p) for p in all_params])

plt.figure(figsize=(max(7, plot_values_len * 0.35), 5))
plt.scatter(ll_vals, ld_vals, alpha=0.8, color="teal")
min_val = np.nanmin([ll_vals, ld_vals])
max_val = np.nanmax([ll_vals, ld_vals])
plt.plot([min_val, max_val], [min_val, max_val], "--", color="gray", linewidth=1)
plt.gca().xaxis.set_major_locator(MaxNLocator(5))
plt.gca().yaxis.set_major_locator(MaxNLocator(5))
plt.xlabel("Mean Slope (LL)")
plt.ylabel("Mean Slope (LD)")
plt.title("Mean Slope Comparison: LL vs LD")
plt.tight_layout()
plt.savefig("mean_slope_scatter_ll_vs_ld.png", dpi=300)
plt.close()
print("✅ Mean slope scatter plot saved as 'mean_slope_scatter_ll_vs_ld.png'")


# 4) Dumbbell plot: LL vs LD mean slope per parameter (sorted by LL)
def plot_dumbbell_mean_slope(params, ll_values, ld_values, out_path):
    data = pd.DataFrame({"Parameter": params, "LL": ll_values, "LD": ld_values}).dropna(
        subset=["LL", "LD"]
    )
    # Match the reference ordering: sort by LL (descending)
    data = data.sort_values("LL", ascending=False).reset_index(drop=True)

    x = np.arange(len(data))
    dynamic_width = min(10.5, max(9.5, len(data) * 0.34))
    fig, ax = plt.subplots(figsize=(dynamic_width, 3.7))
    # lines connecting LL to LD (vertical)
    for i, row in data.iterrows():
        ax.plot([i, i], [row["LL"], row["LD"]], color="gray", linewidth=1, alpha=0.6)
    # points
    ax.scatter(x, data["LL"], color="steelblue", label="LL", zorder=3, s=18)
    ax.scatter(x, data["LD"], color="darkorange", label="LD", zorder=3, s=18)

    ax.set_xticks(x)
    ax.set_xticklabels(data["Parameter"], rotation=90, ha="center", va="top")
    ax.set_ylabel("Mean Slope")
    ax.set_xlabel("Parameter")
    ax.set_title("Mean slope per parameter (LL vs LD)", pad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=True, loc="best", fontsize=9)
    ax.set_xlim(-0.55, len(data) - 0.55)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.52)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✅ Dumbbell plot saved as '{out_path}'")


# Use the mean slope values (LL vs LD) with the original ordering (sorted by LL)
plot_dumbbell_mean_slope(
    all_params,
    ll_vals,
    ld_vals,
    "mean_slope_dumbbell_ll_vs_ld.png",
)
