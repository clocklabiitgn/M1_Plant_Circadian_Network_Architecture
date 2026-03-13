# --- Plot period vs parameter value for each parameter and component ---
import matplotlib.pyplot as plt
import os

# Create output directory for plots
plot_dir = "parameter_fold_change_plots"
os.makedirs(plot_dir, exist_ok=True)

# Reload the delta CSV
import pandas as pd
df = pd.read_csv("parameter_period_deltas.csv")

# Identify parameter/value columns in the delta file
param_col = None
for candidate in ("Parameter", "parameter", "Parameter_Name"):
    if candidate in df.columns:
        param_col = candidate
        break

value_col = None
for candidate in ("Value", "Parameter Value", "Parameter_Value", "param", "parameter"):
    if candidate in df.columns:
        value_col = candidate
        break

if param_col is None or value_col is None:
    raise ValueError(f"Missing required columns. Found columns: {list(df.columns)}")

# Drop rows with missing original period values (i.e., where delta computation failed)
df_clean = df.dropna(subset=['CLm_delta', 'P97m_delta', 'P51m_delta', 'EL_delta'], how='all')

components = ['CLm', 'P97m', 'P51m', 'EL']

# Define base parameter values for fold change calculation
base_values = {
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
    "d6D": 0.576502, "d6L": 0.190108
}

# Compute FoldChange column
df_clean["FoldChange"] = df_clean.apply(
    lambda row: row[value_col] / base_values[row[param_col]] if row[param_col] in base_values else None,
    axis=1
)

# --- Calculate slope of period change vs fold change for each parameter and component ---
from scipy.stats import linregress

# Calculate slope of period change vs fold change for each parameter and component
slope_dict = {}
for param in df_clean[param_col].unique():
    param_data = df_clean[df_clean[param_col] == param]
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
print("Parameter-wise slope of period vs fold change:")
for param, comp_slopes in slope_dict.items():
    print(f"{param}: {comp_slopes}")

# Plot period vs parameter value for each parameter and component
parameters = df_clean[param_col].unique()

# Compute coefficient of variation (CV) for each parameter and component
cv_dict = {}
for param in parameters:
    param_data = df_clean[df_clean[param_col] == param]
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

# Create subplot grid for all components (line plots across parameters)
fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
axes = axes.flatten()
# Okabe-Ito palette (colorblind-friendly)
colors = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#F0E442",  # yellow
    "#000000",  # black
]

for idx, comp in enumerate(components):
    ax = axes[idx]
    for i, param in enumerate(parameters):
        subset = df_clean[df_clean[param_col] == param]
        if subset[comp].notna().any():
            ax.plot(subset["FoldChange"], subset[comp], label=param, marker='o', color=colors[i % len(colors)])
        cv_value = cv_dict[param][comp]
        cv_str = f", CV={cv_value:.2f}" if cv_value is not None else ""
        ax.set_title(f"{comp} Period vs Parameter Value{cv_str}")
    ax.set_xlabel("Fold Change")
    ax.set_ylabel("Period (hours)")
    ax.grid(True)

# Add common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize='small')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("combined_parameter_period_plot.png", dpi=300)
plt.close()

print("✅ Combined period variation plot saved as 'combined_parameter_period_plot.png'")

# --- Combined slope bar plot: Mean slope across all components ---
combined_slope_plot = "combined_mean_slope_bar.png"
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

import pandas as pd
# Save slope values and categories to CSV
slope_df = pd.DataFrame(columns=["Parameter", "Component", "Slope"])
for param, comp_slopes in slope_dict.items():
    for comp, slope_val in comp_slopes.items():
        slope_df = pd.concat([
            slope_df,
            pd.DataFrame([{
                "Parameter": param,
                "Component": comp,
                "Slope": slope_val
            }])
        ], ignore_index=True)

mean_slope_df = pd.DataFrame({
    "Parameter": list(mean_slopes.keys()),
    "MeanSlope": list(mean_slopes.values()),
    "Category": [param_categories[p] for p in mean_slopes.keys()]
})

# Flatten CV dictionary
cv_records = []
for param, comp_cvs in cv_dict.items():
    for comp, cv_val in comp_cvs.items():
        cv_records.append({"Parameter": param, "Component": comp, "CV": cv_val})
cv_df = pd.DataFrame(cv_records)

# Merge with slope summary
summary_df = slope_df.merge(mean_slope_df, on="Parameter", how="left")
summary_df = summary_df.merge(cv_df, on=["Parameter", "Component"], how="left")

summary_df.to_csv("parameter_sensitivity_summary.csv", index=False)
print("✅ Slope values and categories saved to 'parameter_sensitivity_summary.csv'")

# Replace individual slope plots with stacked subplot
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 18), sharex=True)
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
        "Highly Sensitive": "#D55E00",  # vermillion
        "Moderately Sensitive": "#E69F00",  # orange
        "Less Sensitive": "#999999",  # gray
    }
    bar_colors = [color_map.get(param_categories.get(p, "Less Sensitive"), "gray") for p in param_list]
    bars = ax.bar(param_list, slope_list, color=bar_colors)
    for bar in bars:
        height = bar.get_height()
        if height is not None:
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=6, rotation=90)
    ax.set_ylabel(f"Slope ({comp})", fontsize=10)
    ax.grid(axis='x', which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(f"Sensitivity of {comp} Period to Parameter Changes", fontsize=12)
    ax.tick_params(axis='x', labelrotation=90, labelsize=6)

axes[-1].set_xlabel("Parameter", fontsize=12)
plt.tight_layout()
plt.savefig("stacked_component_slope_plots.png", dpi=300)
plt.close()
print("✅ Stacked slope bar plot for all components saved as 'stacked_component_slope_plots.png'")

# Plot combined mean slope bar plot with color by category
sorted_params = sorted(mean_slopes, key=mean_slopes.get, reverse=True)
sorted_slopes = [mean_slopes[p] for p in sorted_params]

color_map = {
    "Highly Sensitive": "#D55E00",  # vermillion
    "Moderately Sensitive": "#E69F00",  # orange
    "Less Sensitive": "#999999",  # gray
}
bar_colors = [color_map[param_categories[p]] for p in sorted_params]

plt.figure(figsize=(14, 6))
plt.bar(sorted_params, sorted_slopes, color=bar_colors)
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Mean Slope (Period vs Fold Change)")
plt.title("Average Sensitivity of Period to Parameter Changes")
plt.tight_layout()
plt.savefig(combined_slope_plot, dpi=300)
plt.close()

print(f"✅ Combined mean slope bar plot saved as '{combined_slope_plot}'")
