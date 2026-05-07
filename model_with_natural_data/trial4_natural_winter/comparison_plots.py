import re
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Helper to parse performance files
#    Expected format (one example line):
#    "CCA1 MAE: 0.5449, MSE: 0.3680"
# -------------------------------------------------------------------

def parse_performance_file(filepath, model_name="Model"):
    """
    Parse a performance txt file and return:
    - metrics: dict[gene] = {"MAE": float, "MSE": float}
    - overall: {"MAE": float, "MSE": float}
    """
    metrics = {}
    overall = {}
    
    mae_pattern = re.compile(r"MAE:\s*([0-9.eE+-]+)")
    mse_pattern = re.compile(r"MSE:\s*([0-9.eE+-]+)")
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Overall line (starts with "MAE:" and "MSE:")
            if line.startswith("MAE:"):
                mae_match = mae_pattern.search(line)
                mse_match = mse_pattern.search(line)
                if mae_match:
                    overall["MAE"] = float(mae_match.group(1))
                if mse_match:
                    overall["MSE"] = float(mse_match.group(1))

            # Per-gene line e.g. "CCA1 MAE: ..., MSE: ..."
            else:
                parts = line.split()
                gene = parts[0]  # first token is gene name
                mae_match = mae_pattern.search(line)
                mse_match = mse_pattern.search(line)
                if mae_match and mse_match:
                    mae_val = float(mae_match.group(1))
                    mse_val = float(mse_match.group(1))
                    metrics[gene] = {"MAE": mae_val, "MSE": mse_val}
    
    return metrics, overall

# -------------------------------------------------------------------
# 2. Read your two models' performance files
#    Adjust filenames / paths if needed.
# -------------------------------------------------------------------

pay_file = "pay_performance.txt"
m1_file = "model1_performance.txt"

pay_metrics, pay_overall = parse_performance_file(pay_file, model_name="Pay")
m1_metrics,  m1_overall  = parse_performance_file(m1_file, model_name="M1")

# For consistent ordering
genes = ["CCA1", "LHY", "PRR9", "PRR7", "PRR5", "TOC1", "ELF4", "LUX"]

# -------------------------------------------------------------------
# 3. Build arrays of MAE and MSE for plotting
# -------------------------------------------------------------------

pay_mae = [pay_metrics[g]["MAE"] for g in genes]
m1_mae  = [m1_metrics[g]["MAE"]  for g in genes]

pay_mse = [pay_metrics[g]["MSE"] for g in genes]
m1_mse  = [m1_metrics[g]["MSE"]  for g in genes]

print("Overall errors:")
print(f"  Pay - MAE: {pay_overall.get('MAE', np.nan):.4f}, MSE: {pay_overall.get('MSE', np.nan):.4f}")
print(f"  M1  - MAE: {m1_overall.get('MAE', np.nan):.4f}, MSE: {m1_overall.get('MSE', np.nan):.4f}")

# -------------------------------------------------------------------
# 4. Grouped bar plot for MAE (main manuscript panel)
# -------------------------------------------------------------------

x = np.arange(len(genes))
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(x - width/2, pay_mae, width, label="Pay model")
ax.bar(x + width/2, m1_mae,  width, label="Model M1")

ax.set_xticks(x)
ax.set_xticklabels(genes, rotation=45, ha="right")
ax.set_ylabel("Mean Absolute Error (MAE)")
ax.set_xlabel("Clock component")
ax.set_title("Per-component MAE: Pay vs Model M1")
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("MAE_comparison_bar.png", dpi=300)
# plt.show()

# -------------------------------------------------------------------
# 5. Scatter plot: MAE(Pay) vs MAE(M1)
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(3, 3))

ax.scatter(pay_mae, m1_mae)

# Add diagonal y = x
all_mae = pay_mae + m1_mae
min_val = min(all_mae)
max_val = max(all_mae)
ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

# Annotate points with gene labels (optional; comment out if too cluttered)
for g, x_val, y_val in zip(genes, pay_mae, m1_mae):
    ax.text(x_val, y_val, g, fontsize=7, ha="left", va="bottom")

ax.set_xlabel("MAE (Pay model)")
ax.set_ylabel("MAE (Model M1)")
ax.set_title("Error comparison per component", fontsize=10)
plt.tight_layout()
plt.savefig("MAE_pay_vs_M1_scatter.png", dpi=300)
# plt.show()

# -------------------------------------------------------------------
# 6. (Optional) Do the same for MSE if you want another panel:
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(pay_mse, m1_mse)

all_mse = pay_mse + m1_mse
min_val_mse = min(all_mse)
max_val_mse = max(all_mse)
ax.plot([min_val_mse, max_val_mse], [min_val_mse, max_val_mse], linestyle="--")

for g, x_val, y_val in zip(genes, pay_mse, m1_mse):
    ax.text(x_val, y_val, g, fontsize=8, ha="left", va="bottom")

ax.set_xlabel("MSE (Pay model)")
ax.set_ylabel("MSE (Model M1)")
ax.set_title("MSE comparison per component")
plt.tight_layout()
plt.savefig("MSE_pay_vs_M1_scatter.png", dpi=300)
# plt.show()
