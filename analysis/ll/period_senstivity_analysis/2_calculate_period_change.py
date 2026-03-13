# Ensure data is loaded from the CSV file
import pandas as pd

data = pd.read_csv("period_data_aggregated.csv")  # Adjust path if necessary

# 11. Calculate period change (delta) for each component per parameter compared to base value
components = ['CLm', 'P97m', 'P51m', 'EL']
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

# Identify parameter/value columns in the aggregated file
param_col = None
for candidate in ("Parameter", "parameter", "Parameter_Name"):
    if candidate in data.columns:
        param_col = candidate
        break

value_col = None
for candidate in ("Value", "Parameter Value", "Parameter_Value", "param", "parameter"):
    if candidate in data.columns:
        value_col = candidate
        break

if param_col is None or value_col is None:
    raise ValueError(f"Missing required columns. Found columns: {list(data.columns)}")

# Create copy and sort to apply delta
delta_df = data.copy().sort_values(by=[param_col, value_col]).reset_index(drop=True)

for comp in components:
    delta_df[f"{comp}_delta"] = delta_df.apply(
        lambda row: row[comp] - delta_df[
            (delta_df[param_col] == row[param_col]) &
            (abs(delta_df[value_col] - base_values.get(row[param_col], -9999)) < 1e-6)
        ][comp].values[0]
        if row[param_col] in base_values and
           ((delta_df[param_col] == row[param_col]) &
            (abs(delta_df[value_col] - base_values[row[param_col]]) < 1e-6)).any()
        else float("nan"),
        axis=1
    )

# Save delta-enhanced table
delta_df.to_csv("parameter_period_deltas.csv", index=False)
print("✅ Period delta table saved as 'parameter_period_deltas.csv'")

