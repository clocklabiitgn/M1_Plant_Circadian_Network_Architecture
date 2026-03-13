import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from math import log10
import pandas as pd
import os
from tqdm import tqdm
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

# Function to simulate the model
def run_simulation(params, t_span=(0, 240), dt=1.0):
    def model(t, C):
        
        ThetaPhyA = ThetaPhyB = ThetaCry1 = 1.0 
        Ired = Iblue = 26.62 

        eta1 = 0.03
        eta2 = 0.0215

        
        dC = np.zeros(20)

        # LHY mRNA
        dC[0] = (params['v1'] + (params['q1a'] * (C[8]) * ThetaPhyA + params['q3a'] * (C[12]) * log10(
            eta1 * Ired + 1) * ThetaPhyB + params['q4a'] * (C[13]) * log10(eta2 * Iblue + 1) * ThetaCry1)) / (
                    1 + (C[3] / params['K1']) ** 2 + (C[5] / params['K2']) ** 2) - (
                    params['k1L'] * ThetaPhyA + params['k1D'] * (1 - ThetaPhyA)) * C[0]
        # LHY protein
        dC[1] = (params['p1'] + params['p1L'] * ThetaPhyA) * C[0] - params['d1'] * C[1]
        # P97 mRNA
        dC[2] = ((params['q1b'] * (C[8]) * ThetaPhyA + params['q3b'] * (C[12]) * log10(
            eta1 * Ired + 1) * ThetaPhyB + params['q4b'] * (C[13]) * log10(eta2 * Iblue + 1) * ThetaCry1) + params[
                    'v2']) * (1 / (1 + (C[1] / params['K3']) ** 2 + (C[5] / params['K4']) ** 2 + (
                    C[7] / params['K5']) ** 2)) - params['k2'] * C[2]
        # P97 protein
        dC[3] = params['p2'] * C[2] - params['d2D'] * (1 - ThetaPhyA) * C[3] - params['d2L'] * ThetaPhyA * C[3]
        # P51 mRNA
        dC[4] = params['v3'] / (1 + (C[1] / params['K6']) ** 2 + (C[5] / params['K7']) ** 2 + (
                    C[19] / params['K13']) ** 2) - params['k3'] * C[4]
        # P51 protein
        dC[5] = (params['p3'] * C[4] - params['d3D'] * (1 - ThetaPhyA) * C[5] - params['d3L'] * ThetaPhyA * C[5])
        # EL mRNA
        dC[6] = (params['v4'] * ThetaPhyA / (1 + (C[1] / params['K8']) ** 2 + (C[5] / params['K9']) ** 2 + (
                    C[7] / params['K10']) ** 2) - params['k4'] * C[6])
        # EL protein
        dC[7] = (params['p4'] * C[6] - (params['de1'] + (
                    params['de2'] * C[14] + params['de3'] * C[15] + params['de4'] * C[16] + params['de5'] * C[17]) / (
                                                C[14] + C[15] + C[16] + C[17])) * C[7])
        # PHY A
        dC[8] = (1 - ThetaPhyA) * params['Ap3'] - (params['Am7'] * C[8] / (params['Ak7'] + C[8])) - params['q2'] * ThetaPhyA * \
                C[8] - params['kmpac'] * ThetaPhyA * C[8] * C[14] + params['kd'] * (1 - ThetaPhyA) * C[15]
        # PIF mRNA
        dC[9] = params['v5'] / (1 + (C[7] / params['K11']) ** 2 + (C[13] / params['K14']) ** 2) - params['k5'] * C[9]
        # PIF protein
        dC[10] = params['p5'] * C[9] - params['d5D'] * (1 - ThetaPhyA) * C[10] - params['d5L'] * ThetaPhyA * C[10]
        # HYP protein
        dC[11] = params['g1'] + (params['g2'] * C[10] ** 2) / (params['K12'] ** 2 + C[10] ** 2)
        # PHY B
        dC[12] = params['Bp4'] - ((params['Bm8'] * C[12]) / (params['Bk8'] + C[12])) - params['kmpbc'] * ThetaPhyB * C[12] * \
                C[14] + params['kd'] * (1 - ThetaPhyB) * C[16]
        # CRY1
        dC[13] = params['Cp5'] - ((params['Cm9'] * C[13]) / (params['Ck9'] + C[13])) - params['kmcc'] * ThetaCry1 * C[13] * \
                C[14] + params['kd'] * (1 - ThetaCry1) * C[17]
        # COP1
        dC[14] = -params['kmpac'] * ThetaPhyA * C[8] * C[14] + params['kd'] * (1 - ThetaPhyA) * C[15] - params[
            'kmpbc'] * ThetaPhyB * C[12] * C[14] + params['kd'] * (1 - ThetaPhyB) * C[16] - params['kmcc'] * ThetaCry1 * C[
                    13] * C[14] + params['kd'] * (1 - ThetaCry1) * C[17] + (
                        params['Am7'] * C[15] / (params['Ak7'] + C[15])) + params['q2'] * ThetaPhyA * C[15] + (
                        (params['Bm8'] * C[16]) / (params['Bk8'] + C[16])) + (
                        (params['Cm9'] * C[17]) / (params['Ck9'] + C[17]))
        # COP1:PHYA
        dC[15] = params['kmpac'] * ThetaPhyA * C[8] * C[14] - params['kd'] * (1 - ThetaPhyA) * C[15] - (
                    params['Am7'] * C[15] / (params['Ak7'] + C[15])) - params['q2'] * ThetaPhyA * C[15]
        # COP1:PHYB
        dC[16] = params['kmpbc'] * ThetaPhyB * C[12] * C[14] - params['kd'] * (1 - ThetaPhyB) * C[16] - (
                    (params['Bm8'] * C[16]) / (params['Bk8'] + C[16]))
        # COP1:CRY1
        dC[17] = params['kmcc'] * ThetaCry1 * C[13] * C[14] - params['kd'] * (1 - ThetaCry1) * C[17] - (
                    (params['Cm9'] * C[17]) / (params['Ck9'] + C[17]))
        # GZ mRNA
        dC[18] = params['v6'] / (1 + (C[14] / params['K15']) ** 2) - params['k6'] * C[18]
        # GZ protein
        dC[19] = (params['p6'] * C[18] - params['d6D'] * (1 - ThetaPhyA) * C[19] - params['d6L'] * ThetaPhyA * C[19])

        return dC
    
    C0 = np.ones(20)
    C0[10] = C0[15] = C0[16] = C0[17] = 0 # Reset initial values
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(model, t_span, C0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.t, sol.y

# Function to compute period
def compute_period(time, signal):
    # Exclude the first 72 data points
    start_index = 72
    time = time[start_index:]
    signal = signal[start_index:]
    
    # Normalize the signal
    normalized_signal = signal / np.max(signal)
    
    # Check if the signal has sufficient amplitude variation
    amplitude_threshold = 0.1  # Minimum amplitude variation required
    if np.max(normalized_signal) - np.min(normalized_signal) < amplitude_threshold:
        # Return NaN if the signal is too flat
        return np.nan
    
    # Find peaks in the normalized signal
    peaks, _ = find_peaks(normalized_signal)
    
    # Ensure there are enough peaks to calculate a meaningful period
    if len(peaks) > 2:
        periods = np.diff(time[peaks])
        return np.mean(periods)
    
    # Return NaN if insufficient peaks are detected
    return np.nan

def normalize_signal(signal):
    max_val = np.max(signal)
    if max_val == 0:
        return np.zeros_like(signal)
    return signal / max_val

# Running knockout analysis
params = define_parameters()
parameter_names = list(params.keys())
results = []
outfile = "knockout_results/knockout_analysis.csv"
os.makedirs("knockout_results", exist_ok=True)
profiles_dir = "knockout_results/expression_profiles"
plots_dir = "knockout_results/plots"
os.makedirs(profiles_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

with open(outfile, "w") as f:
    f.write("Parameter,Value,Period_0,Period_2,Period_4,Period_6\n")

for param in tqdm(parameter_names, desc="Running Knockout Analysis"):
    base_value = params[param]
    profile_cache = {}
    for value, label in [(base_value, "baseline"), (0, "knockout")]:  # Original and knockout
        params = define_parameters()
        safe_value = value if value != 0 else 1e-6  # Avoid division by zero
        params[param] = safe_value
        time, C = run_simulation(params)
        period_values = [compute_period(time, C[i]) for i in [0, 2, 4, 6]]
        results.append([param, value] + period_values)
        
        with open(outfile, "a") as f:
            f.write(f"{param},{value},{period_values[0]},{period_values[1]},{period_values[2]},{period_values[3]}\n")

        start_index = 72
        cropped_time = time[start_index:] - time[start_index]
        raw_profiles = {
            "CLm": C[0][start_index:],
            "P97m": C[2][start_index:],
            "P51m": C[4][start_index:],
            "EL": C[6][start_index:],
        }
        norm_profiles = {name: normalize_signal(signal) for name, signal in raw_profiles.items()}
        profile_cache[label] = (value, cropped_time, norm_profiles)

        profiles_outfile = os.path.join(profiles_dir, f"{param}_{label}.csv")
        profile_df = pd.DataFrame({
            "Time": cropped_time,
            "CLm_raw": raw_profiles["CLm"],
            "P97m_raw": raw_profiles["P97m"],
            "P51m_raw": raw_profiles["P51m"],
            "EL_raw": raw_profiles["EL"],
            "CLm_norm": norm_profiles["CLm"],
            "P97m_norm": norm_profiles["P97m"],
            "P51m_norm": norm_profiles["P51m"],
            "EL_norm": norm_profiles["EL"],
        })
        profile_df.to_csv(profiles_outfile, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    species = [("CLm", (0, 0)), ("P97m", (0, 1)), ("P51m", (1, 0)), ("EL", (1, 1))]
    for name, (row, col) in species:
        ax = axes[row, col]
        for label in ["baseline", "knockout"]:
            value, time, norm_profiles = profile_cache[label]
            ax.plot(time, norm_profiles[name], label=f"{param}={value}")
        ax.set_title(name)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized expression")
    fig.suptitle(f"Expression profiles across {param} (norm)", y=0.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    plot_outfile = os.path.join(plots_dir, f"{param}_norm_profiles.png")
    fig.savefig(plot_outfile, dpi=300)
    plt.close(fig)
