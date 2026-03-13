import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from math import log10
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import sys
from itertools import combinations

def define_parameters():
    params = {
        "v1": 4.8318, "q1a": 1.4266, "q3a": 8.9432, "q4a": 5.9277, 
        "K1": 0.1943, "K2": 1.6138, "k1L": 0.2866,  "p1": 0.8672, "p1L": 0.2378,
        "d1": 0.7843, "q1b": 3.575, "q3b": 5.5899, "q4b": 8.954, "v2": 1.6822, "K3": 2.2275,
        "K4": 0.40, "K5": 0.37, "k2": 0.35, "p2": 0.7858, "d2L": 0.2917,
        "v3": 1.113, "K6": 0.4944, "K7": 2.4087, "k3": 0.5819, "p3": 0.6142, 
        "d3L": 0.5431, "v4": 2.5012, "K8": 0.3262, "K9": 1.7974, "K10": 1.1889, "k4": 0.925,
        "p4": 1.126, "de1": 0.0022, 
        "Ap3": 0.3868, "Am7": 0.5503, "Ak7": 1.125,  "kmpac": 137, "kd": 7,
        "v5": 0.1129, "K11": 0.3322, "k5": 0.1591, "p5": 0.5293,  "d5L": 5.0712,
        "g1": 0.001, "g2": 0.18, "K12": 0.86, "Bp4": 0.4147, "Bm8": 0.7728, "Bk8": 0.1732,
        "kmpbc": 7162, "Cp5": 0.4567, "Cm9": 0.867, "Ck9": 0.3237, "kmcc": 13406, "K13": 8.04866,
        "K14": 0.347569, "v6": 0.438785, "k6": 0.328256, "K15": 0.336558, "p6": 0.264064,
        "d6L": 0.190108, "de2": 0.4741, "de3": 0.3765, "de4": 0.398, "de5": 0.0003, "q2": 0.5767,
        "k1D": 0.213, "d2D": 0.3712, "d3D": 0.5026, "d5D": 0.4404, "d6D": 1.0
    }
    return params

def run_simulation(params, t_span=(0, 240), dt=1.0):
    def model(t, C):
        #light conditions = constant light (LL) with 26.62 μmol m^-2 s^-1 of red and blue light, and 1.0 for phyA, phyB, and cry1 activation levels
        ThetaPhyA = ThetaPhyB = ThetaCry1 = 1.0 
        Ired = Iblue = 26.62
        eta1, eta2 = 0.03, 0.0215
        


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
        dC[5] = params['p3'] * C[4] - params['d3D'] * (1 - ThetaPhyA) * C[5] - params['d3L'] * ThetaPhyA * C[5]

        # EL mRNA
        dC[6] = params['v4'] * ThetaPhyA / (1 + (C[1] / params['K8']) ** 2 + (C[5] / params['K9']) ** 2 + (
            C[7] / params['K10']) ** 2) - params['k4'] * C[6]

        # EL protein
        dC[7] = params['p4'] * C[6] - (params['de1'] + (
            params['de2'] * C[14] + params['de3'] * C[15] + params['de4'] * C[16] + params['de5'] * C[17]) / (
            C[14] + C[15] + C[16] + C[17])) * C[7]

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
        dC[19] = params['p6'] * C[18] - params['d6D'] * (1 - ThetaPhyA) * C[19] - params['d6L'] * ThetaPhyA * C[19]

        return dC

    C0 = np.ones(20)
    C0[10] = C0[15] = C0[16] = C0[17] = 0  # Reset initial values
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(model, t_span, C0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.t, sol.y


def compute_period(time, signal):
    start_index = 72
    time = time[start_index:]
    signal = signal[start_index:]

    normalized_signal = signal / np.max(signal)

    amplitude_threshold = 0.02
    if np.max(normalized_signal) - np.min(normalized_signal) < amplitude_threshold:
        # print("Flat signal detected, skipping period calculation.")
        return np.nan

    # Try gentler peak detection
    peaks, properties = find_peaks(normalized_signal, height=0.1, prominence=0.01)

    # print(f"→ Detected {len(peaks)} peaks at indices: {peaks}")
    if len(peaks) >= 3:
        periods = np.diff(time[peaks])
        # print(f"→ Periods between peaks: {periods}")
        return np.mean(periods)

    # print("→ Not enough peaks detected")
    return np.nan


def adaptive_param_range(base_value, step_factor=1.1, max_fold=10):
    """Symmetric around base_value: e.g., 1x, 1.1x, 0.9x, 1.21x, 0.81x, ..."""
    visited = set()
    yield base_value
    visited.add(round(base_value, 8))

    i = 1
    while True:
        up = round(base_value * (step_factor ** i), 8)
        down = round(base_value / (step_factor ** i), 8)

        if up / base_value > max_fold and base_value / down > max_fold:
            break

        if up not in visited:
            yield up
            visited.add(up)
        if down not in visited:
            yield down
            visited.add(down)
        i += 1



def analyze_parameter(param, base_value, components, results_dir):
    try:
        print(f"Analyzing parameter: {param} with base value: {base_value}")
        param_dir = os.path.join(results_dir, param)
        os.makedirs(param_dir, exist_ok=True)

        component_names = {0: 'CLm', 2: 'P97m', 4: 'P51m', 6: 'EL'}

        param_gen = adaptive_param_range(base_value, step_factor=1.1, max_fold=50)
        loss_recorded = False
        period_data = []
        period_csv = os.path.join(param_dir, f"period_data_{param}.csv")

        previous_periods = None
        no_change_count = 0
        no_change_limit = 50

        for val in param_gen:
            # print(f"  → Testing {param} = {val:.6f}")
            params = define_parameters()
            params[param] = val
            time, simulated_data = run_simulation(params)

            start_index = 72
            time = time[start_index:]
            simulated_data = simulated_data[:, start_index:]

            # Save expression profiles for this single run (raw + normalized)
            expr_out = pd.DataFrame({"Time": time})
            for comp in components:
                name = component_names[comp]
                y = simulated_data[comp].astype(float)
                expr_out[f"{name}_raw"] = y
                max_y = np.max(y)
                expr_out[f"{name}_norm"] = (y / max_y) if max_y != 0 else np.nan

            expr_csv = os.path.join(param_dir, f"expression_profile_{param}_{val:.6f}.csv")
            expr_out.to_csv(expr_csv, index=False)

            periods = [compute_period(time, simulated_data[i]) for i in components]

            if previous_periods is not None and all(
                np.isclose(p, pp, atol=0.1, equal_nan=True) for p, pp in zip(periods, previous_periods)):
                no_change_count += 1
            else:
                no_change_count = 0
            previous_periods = periods.copy()

            if no_change_count >= no_change_limit:
                print(f"→ Periods unchanged for {no_change_limit} consecutive values. Stopping scan for {param}.")
                break

            for i, period in enumerate(periods):
                if np.isnan(period):
                    comp = components[i]
                    plt.figure()
                    plt.plot(time, simulated_data[comp])
                    plt.title(f"{component_names[comp]} trace (NaN period)")
                    plt.xlabel("Time")
                    plt.ylabel("Expression")
                    plt.savefig(os.path.join(param_dir, f"debug_trace_{component_names[comp]}_{val:.6f}.png"))
                    plt.close()

            row = [val] + periods
            period_data.append(row)
            df_append = pd.DataFrame([row], columns=["Parameter Value"] + [component_names[i] for i in components])
            if not os.path.exists(period_csv):
                df_append.to_csv(period_csv, index=False)
            else:
                df_append.to_csv(period_csv, index=False, mode='a', header=False)

            # Expression Plots
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Expression Plots for {param} = {val:.6f}")

            for i, comp in enumerate(components):
                normalized_data = simulated_data[comp] / np.max(simulated_data[comp])
                row_idx, col_idx = divmod(i, 2)
                axs[row_idx, col_idx].plot(time, normalized_data)
                axs[row_idx, col_idx].set_title(component_names[comp])
                axs[row_idx, col_idx].set_xlabel("Time")
                axs[row_idx, col_idx].set_ylabel("Expression Level")

            plt.tight_layout()
            plt.savefig(os.path.join(param_dir, f"expression_plot_{param}_{val:.6f}.png"))
            plt.close()

            # Phase Portraits
            num_components = len(components)
            component_pairs = list(combinations(components, 2))
            num_plots = len(component_pairs)
            grid_size = int(np.ceil(np.sqrt(num_plots)))

            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 10))
            axes = axes.flatten()
            fig.suptitle(f"Phase Portraits for {param} = {val:.6f}")

            for i, (comp1, comp2) in enumerate(component_pairs):
                axes[i].plot(simulated_data[comp1], simulated_data[comp2], label=f'{component_names[comp1]} vs {component_names[comp2]}')
                axes[i].set_xlabel(component_names[comp1])
                axes[i].set_ylabel(component_names[comp2])
                axes[i].legend()

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.savefig(os.path.join(param_dir, f"phase_portraits_{param}_{val:.6f}.png"))
            plt.close()

            if all(np.isnan(p) or p == 0 for p in periods):
                if not loss_recorded:
                    with open(os.path.join(param_dir, "loss_threshold.txt"), "w") as f:
                        f.write(f"Periodicity lost at param value = {val:.6f}\n")
                    loss_recorded = True
                continue

        print(f"✓ Completed analysis for parameter: {param}")
        return period_data
    except Exception as e:
        error_path = os.path.join(results_dir, f"error_{param}.log")
        with open(error_path, "w") as f:
            f.write(str(e))
        return []


def main():
    mp.set_start_method('spawn', force=True)
    params = define_parameters()
    components = [0, 2, 4, 6]
    results_dir = "parameter_analysis"
    os.makedirs(results_dir, exist_ok=True)

    with mp.Pool(processes=60) as pool:
        tasks = [(param, base_value, components, results_dir)
                 for param, base_value in params.items()]
        list(tqdm(pool.starmap(analyze_parameter, tasks), total=len(tasks), desc="Parameters Completed"))

    print("Analysis completed and results saved.")

    


if __name__ == "__main__":
    main()
