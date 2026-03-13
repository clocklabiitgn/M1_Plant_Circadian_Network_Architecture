import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use script location for all input/output paths (no CLI args needed).
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "parameter_analysis"

# Component names
components = ["CLm", "P97m", "P51m", "EL"]

# Summary storage
summary_data = []
heatmap_data = []
aggregated_frames = []

# Iterate through parameter folders
for folder in sorted(DATA_DIR.glob('*/')):
    if not folder.is_dir():
        continue

    param_name = folder.name
    csv_files = list(folder.glob("period_data_*.csv"))
    if not csv_files:
        print(f"No period data found in {param_name}")
        continue

    csv_path = csv_files[0]  # assuming only one CSV file per folder
    df = pd.read_csv(csv_path)

    param_col = None
    for candidate in ("param", "Parameter Value", "Parameter_Value", "parameter", "Parameter"):
        if candidate in df.columns:
            param_col = candidate
            break
    if param_col is None:
        print(f"Missing parameter column in {csv_path.name}")
        continue

    df_with_meta = df.copy()
    df_with_meta["Parameter"] = param_name
    df_with_meta["Source CSV"] = csv_path.name
    aggregated_frames.append(df_with_meta)

    component_stats = {}
    heatmap_row = {'Parameter': param_name}

    for comp in components:
        if comp not in df.columns:
            component_stats[comp] = 'Missing'
            heatmap_row[comp] = np.nan
            continue

        period_values = df[comp].dropna()
        if period_values.empty:
            component_stats[comp] = 'No Rhythmicity'
            heatmap_row[comp] = np.nan
            continue

        # Period variation
        period_range = period_values.max() - period_values.min()
        percent_variation = 100 * period_range / period_values.mean()

        heatmap_row[comp] = percent_variation  # for heatmap

        # Sensitivity classification
        if percent_variation > 20:
            level = 'Highly sensitive'
        elif percent_variation > 5:
            level = 'Moderately sensitive'
        else:
            level = 'Insensitive'

        component_stats[comp] = f"{level} ({percent_variation:.2f}%)"

        # Plotting
        plt.figure()
        plt.plot(df[param_col], df[comp], marker='o')
        plt.title(f"{param_name} effect on {comp}")
        plt.xlabel(param_col)
        plt.ylabel('Period (h)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(folder / f"period_plot_{comp}.png")
        plt.close()

    # Append summary
    row = {'Parameter': param_name}
    row.update(component_stats)
    summary_data.append(row)
    heatmap_data.append(heatmap_row)

# Save detailed summary
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(BASE_DIR / 'parameter_period_sensitivity_summary.csv', index=False)

# Create and save heatmap
heatmap_df = pd.DataFrame(heatmap_data).set_index('Parameter')
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': '% Variation'})
plt.title("Period Sensitivity Heatmap (% variation)")
plt.ylabel("Parameter")
plt.xlabel("Component")
plt.tight_layout()
plt.savefig(BASE_DIR / "period_sensitivity_heatmap.png")
plt.close()

# Rank parameters by total influence
heatmap_df['Total Influence'] = heatmap_df[components].abs().sum(axis=1)
ranked_df = heatmap_df[['Total Influence']].sort_values(by='Total Influence', ascending=False)
ranked_df.to_csv(BASE_DIR / 'parameter_influence_ranking.csv')

if aggregated_frames:
    aggregated_df = pd.concat(aggregated_frames, ignore_index=True)
    aggregated_df.to_csv(BASE_DIR / "period_data_aggregated.csv", index=False)

print("✅ Analysis complete.\n- Summary: parameter_period_sensitivity_summary.csv\n- Heatmap: period_sensitivity_heatmap.png\n- Ranking: parameter_influence_ranking.csv\n- Aggregated: period_data_aggregated.csv")
