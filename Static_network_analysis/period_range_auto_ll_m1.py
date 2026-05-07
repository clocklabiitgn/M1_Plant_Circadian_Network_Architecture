"""
period_range_auto_ll_m1.py
--------------------------
Adaptive parameter-range scan for the M1 circadian clock model under
constant-light (LL) conditions.

Changes from the original period_range_auto_ld.py
--------------------------------------------------
* 19-state ODE system matching m1.py (GZ protein only, no separate mRNA).
* GZ parameters: Gp6, dg1-dg5, dp6, Gkp  (removed v6, k6, K15, p6, d6D, d6L).
* P51 protein (Eq 5): GZ-mediated Michaelis degradation term added.
* CL/P97 mRNA light terms use C[8]+C[15] and C[12]+C[16].
* P51 mRNA (Eq 4): K6, K7 only (K13 term removed).
* LL condition: ThetaPhyA = ThetaPhyB = ThetaCry1 = 1, Ired = Iblue = 26.62 (constant).
* Integration: same odeint hour-by-hour loop as m1.py / LD script (Nday=20, 480 h total).
* eta1, eta2 excluded from scan (not biologically meaningful to vary).
* Pool size capped at min(8, cpu_count) — avoids oversubscription on 16-core workstation.
* Output directory: parameter_analysis_ll/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
from math import log
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Parameters — from m1.py
# ---------------------------------------------------------------------------
def define_parameters():
    return {
        # CL (LHY/CCA1)
        "v1":    4.8318,   "q1a":  1.4266,  "q3a":  8.9432,  "q4a":  5.9277,
        "K1":    0.1943,   "K2":   1.6138,  "k1L":  0.2866,  "k1D":  0.213,
        "p1":    0.8672,   "p1L":  0.2378,  "d1":   0.7843,
        # P97 (PRR9/PRR7)
        "q1b":   3.575,    "q3b":  5.5899,  "q4b":  8.954,   "v2":   1.6822,
        "K3":    2.2275,   "K4":   0.40,    "K5":   0.37,    "k2":   0.35,
        "p2":    0.7858,   "d2D":  0.3712,  "d2L":  0.2917,
        # P51 (PRR5/TOC1)
        "v3":    1.113,    "K6":   0.4944,  "K7":   2.4087,  "k3":   0.5819,
        "p3":    0.6142,   "d3D":  0.5026,  "d3L":  0.5431,
        # EL (ELF4/LUX)
        "v4":    2.5012,   "K8":   0.3262,  "K9":   1.7974,  "K10":  1.1889,
        "k4":    0.925,    "p4":   1.126,
        "de1":   0.0022,   "de2":  0.4741,  "de3":  0.3765,
        "de4":   0.398,    "de5":  0.0003,
        # PhyA
        "Ap3":   0.3868,   "Am7":  0.5503,  "Ak7":  1.125,   "q2":   0.5767,
        "kmpac": 137,      "kd":   7,
        # PIF
        "v5":    0.1129,   "K11":  0.3322,  "K14":  1.5,     "k5":   0.1591,
        "p5":    0.5293,   "d5D":  0.4404,  "d5L":  5.0712,
        # Hypocotyl
        "g1":    0.001,    "g2":   0.18,    "K12":  0.86,
        # PhyB
        "Bp4":   0.4147,   "Bm8":  0.7728,  "Bk8":  0.1732,  "kmpbc": 7162,
        # Cry1
        "Cp5":   0.4567,   "Cm9":  0.867,   "Ck9":  0.3237,  "kmcc": 13406,
        # GZ (direct protein dynamics)
        "Gp6":   0.000100, "dg1":  0.010000,"dg2":  1.280202,
        "dg3":   0.010000, "dg4":  1.750462,"dg5":  1.067661,
        "dp6":   0.010000, "Gkp":  1.185527,
        # Light normalisation (excluded from scan)
        "eta1":  0.03,     "eta2": 0.0215,
    }

# ---------------------------------------------------------------------------
# ODE RHS factory — LL: flags and intensities fixed, odeint-compatible
# ---------------------------------------------------------------------------
def model(params):
    """Returns rhs(C, t) for odeint — constant light throughout."""
    ThetaPhyA = ThetaPhyB = ThetaCry1 = 1.0
    Ired = Iblue = 26.62
    eta1 = params["eta1"]
    eta2 = params["eta2"]

    def model(C, t):
        cop1_sum = C[14] + C[15] + C[16] + C[17]
        if cop1_sum == 0:
            cop1_sum = 1e-12
        dC = np.zeros(19)

        # Eq 0 — LHY mRNA
        dC[0] = (
            params["v1"]
            + (  params["q1a"] * (C[8]) * ThetaPhyA
               + params["q3a"] * (C[12]) * log(eta1 * Ired  + 1) * ThetaPhyB
               + params["q4a"] * (C[13]) * log(eta2 * Iblue + 1) * ThetaCry1)
        ) / (1 + (C[3] / params["K1"])**2 + (C[5] / params["K2"])**2) \
          - (params["k1L"] * ThetaPhyA + params["k1D"] * (1 - ThetaPhyA)) * C[0]

        # Eq 1 — LHY protein
        dC[1] = (params["p1"] + params["p1L"] * ThetaPhyA) * C[0] - params["d1"] * C[1]

        # Eq 2 — P97 mRNA
        dC[2] = (
            (  params["q1b"] * (C[8]) * ThetaPhyA
             + params["q3b"] * (C[12]) * log(eta1 * Ired  + 1) * ThetaPhyB
             + params["q4b"] * (C[13]) * log(eta2 * Iblue + 1) * ThetaCry1)
            + params["v2"]
        ) * (1 / (1 + (C[1] / params["K3"])**2 + (C[5] / params["K4"])**2
                    + (C[7] / params["K5"])**2)) \
          - params["k2"] * C[2]

        # Eq 3 — P97 protein
        dC[3] = params["p2"] * C[2] \
                - params["d2D"] * (1 - ThetaPhyA) * C[3] \
                - params["d2L"] * ThetaPhyA        * C[3]

        # Eq 4 — P51 mRNA (K6, K7 only)
        dC[4] = params["v3"] / (1 + (C[1] / params["K6"])**2 + (C[5] / params["K7"])**2) \
                - params["k3"] * C[4]

        # Eq 5 — P51 protein (GZ-mediated Michaelis degradation)
        dC[5] = params["p3"] * C[4] \
                - params["d3D"] * (1 - ThetaPhyA) * C[5] \
                - params["d3L"] * ThetaPhyA        * C[5] \
                - (params["dp6"] * C[18] * C[5]) / (params["Gkp"] + C[5])

        # Eq 6 — EL mRNA
        dC[6] = params["v4"] * ThetaPhyA \
                / (1 + (C[1] / params["K8"])**2 + (C[5] / params["K9"])**2
                     + (C[7] / params["K10"])**2) \
                - params["k4"] * C[6]

        # Eq 7 — EL protein
        dC[7] = params["p4"] * C[6] \
                - (params["de1"]
                   + (params["de2"] * C[14] + params["de3"] * C[15]
                      + params["de4"] * C[16] + params["de5"] * C[17]) / cop1_sum) * C[7]

        # Eq 8 — PhyA
        dC[8] = (1 - ThetaPhyA) * params["Ap3"] \
                - (params["Am7"] * C[8] / (params["Ak7"] + C[8])) \
                - params["q2"]    * ThetaPhyA * C[8] \
                - params["kmpac"] * ThetaPhyA * C[8] * C[14] \
                + params["kd"] * C[15]

        # Eq 9 — PIF mRNA
        dC[9] = params["v5"] / (1 + (C[7] / params["K11"])**2
                                   + (C[13] / params["K14"])**2) \
                - params["k5"] * C[9]

        # Eq 10 — PIF protein
        dC[10] = params["p5"] * C[9] \
                 - params["d5D"] * (1 - ThetaPhyA) * C[10] \
                 - params["d5L"] * ThetaPhyA        * C[10]

        # Eq 11 — HYP protein
        dC[11] = params["g1"] + (params["g2"] * C[10]**2) / (params["K12"]**2 + C[10]**2)

        # Eq 12 — PhyB
        dC[12] = params["Bp4"] \
                 - (params["Bm8"] * C[12] / (params["Bk8"] + C[12])) \
                 - params["kmpbc"] * ThetaPhyB * C[12] * C[14] \
                 + params["kd"] * C[16]

        # Eq 13 — Cry1
        dC[13] = params["Cp5"] \
                 - (params["Cm9"] * C[13] / (params["Ck9"] + C[13])) \
                 - params["kmcc"] * ThetaCry1 * C[13] * C[14] \
                 + params["kd"] * C[17]

        # Eq 14 — COP1
        dC[14] = (
            - params["kmpac"]  * ThetaPhyA * C[8]  * C[14] + params["kd"] * C[15]
            - params["kmpbc"]  * ThetaPhyB * C[12] * C[14] + params["kd"] * C[16]
            - params["kmcc"]   * ThetaCry1 * C[13] * C[14] + params["kd"] * C[17]
            + (params["Am7"] * C[15] / (params["Ak7"] + C[15]))
            +  params["q2"]   * ThetaPhyA * C[15]
            + (params["Bm8"] * C[16] / (params["Bk8"] + C[16]))
            + (params["Cm9"] * C[17] / (params["Ck9"] + C[17]))
        )

        # Eq 15 — COP1:PhyA
        dC[15] = params["kmpac"] * ThetaPhyA * C[8]  * C[14] \
                 - params["kd"] * C[15] \
                 - (params["Am7"] * C[15] / (params["Ak7"] + C[15])) \
                 - params["q2"] * ThetaPhyA * C[15]

        # Eq 16 — COP1:PhyB
        dC[16] = params["kmpbc"] * ThetaPhyB * C[12] * C[14] \
                 - params["kd"] * C[16] \
                 - (params["Bm8"] * C[16] / (params["Bk8"] + C[16]))

        # Eq 17 — COP1:Cry1
        dC[17] = params["kmcc"] * ThetaCry1 * C[13] * C[14] \
                 - params["kd"] * C[17] \
                 - (params["Cm9"] * C[17] / (params["Ck9"] + C[17]))

        # Eq 18 — GZ protein
        dC[18] = params["Gp6"] \
                 - (params["dg1"]
                    + (params["dg2"] * C[14] + params["dg3"] * C[15]
                       + params["dg4"] * C[16] + params["dg5"] * C[17]) / cop1_sum) * C[18]

        return dC

    return model


# ---------------------------------------------------------------------------
# Simulation — same odeint hourly loop as m1.py / LD script
# ---------------------------------------------------------------------------
def run_simulation(params, Nday=20):
    """
    Integrate M1 under constant-light (LL) using the same hour-by-hour
    odeint loop as m1.py.  Ired = Iblue = 26.62 every hour.

    Returns (time, C) where C.shape == (19, Nday*24 + 1).
    """
    rhs = model(params)

    C0 = np.ones(19)
    C0[11] = C0[15] = C0[16] = C0[17] = 0.0

    C_all = [C0.copy()]
    for t in range(Nday * 24):
        sol = odeint(rhs, C_all[-1], [t, t + 1])
        C_all.append(sol[-1])

    C_arr = np.array(C_all).T          # shape (19, Nday*24 + 1)
    time  = np.arange(C_arr.shape[1], dtype=float)
    return time, C_arr


# ---------------------------------------------------------------------------
# Period estimation — discard first 240 h as transient
# ---------------------------------------------------------------------------
def compute_period(time, signal, burn_in=240):
    idx = np.searchsorted(time, burn_in)
    t = time[idx:]
    s = signal[idx:]

    s_range = np.max(s) - np.min(s)
    if s_range < 0.02 * (np.max(np.abs(s)) + 1e-12):
        return np.nan

    s_norm = (s - np.min(s)) / (s_range + 1e-12)
    peaks, _ = find_peaks(s_norm, height=0.1, prominence=0.05)
    if len(peaks) >= 3:
        return float(np.mean(np.diff(t[peaks])))
    return np.nan


# ---------------------------------------------------------------------------
# Adaptive parameter range generator (unchanged from original)
# ---------------------------------------------------------------------------
def adaptive_param_range(base_value, step_factor=1.1, max_fold=10):
    """Yield base, then alternating up/down multiplicative steps."""
    visited = set()
    yield base_value
    visited.add(round(base_value, 8))
    i = 1
    while True:
        up   = round(base_value * (step_factor ** i), 8)
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


# ---------------------------------------------------------------------------
# Per-parameter analysis worker
# ---------------------------------------------------------------------------
COMPONENT_NAMES = {0: "LHYm", 2: "P97m", 4: "P51m", 6: "ELm"}
COMPONENTS      = [0, 2, 4, 6]

def analyze_parameter(args):
    param, base_value, results_dir = args
    try:
        print(f"  Analysing: {param}  (base = {base_value})")
        param_dir = os.path.join(results_dir, param)
        os.makedirs(param_dir, exist_ok=True)

        period_csv   = os.path.join(param_dir, f"period_data_{param}.csv")
        loss_recorded = False
        previous_periods   = None
        no_change_count    = 0
        NO_CHANGE_LIMIT    = 50
        col_names = ["Parameter_Value"] + [COMPONENT_NAMES[i] for i in COMPONENTS]

        for val in adaptive_param_range(base_value, step_factor=1.1, max_fold=10):
            p = define_parameters()
            p[param] = val

            try:
                time, C = run_simulation(p)
            except Exception as e:
                print(f"    [warn] {param}={val:.4g}: solver failed — {e}")
                continue

            # Compute periods
            periods = [compute_period(time, C[i]) for i in COMPONENTS]

            # Early-stop if periods are stable
            if previous_periods is not None and all(
                    np.isclose(p_new, p_old, atol=0.1, equal_nan=True)
                    for p_new, p_old in zip(periods, previous_periods)):
                no_change_count += 1
            else:
                no_change_count = 0
            previous_periods = periods.copy()

            if no_change_count >= NO_CHANGE_LIMIT:
                print(f"    Stable for {NO_CHANGE_LIMIT} steps — stopping scan for {param}.")
                break

            # --- Save expression profiles CSV ---
            burn_idx = np.searchsorted(time, 240)
            t_crop   = time[burn_idx:]
            expr_df  = pd.DataFrame({"Time": t_crop})
            for comp in COMPONENTS:
                name = COMPONENT_NAMES[comp]
                y    = C[comp][burn_idx:].astype(float)
                expr_df[f"{name}_raw"]  = y
                mx = np.max(np.abs(y))
                expr_df[f"{name}_norm"] = y / mx if mx > 0 else np.nan
            expr_df.to_csv(
                os.path.join(param_dir, f"expr_{param}_{val:.6f}.csv"),
                index=False)

            # --- Append period row ---
            row = [val] + periods
            df_row = pd.DataFrame([row], columns=col_names)
            write_header = not os.path.exists(period_csv)
            df_row.to_csv(period_csv, mode='a', header=write_header, index=False)

            # --- Expression plot (2×2) ---
            fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
            fig.suptitle(f"LL — {param} = {val:.4g}", fontsize=11)
            for idx, comp in enumerate(COMPONENTS):
                ax   = axs[idx // 2][idx % 2]
                y    = C[comp][burn_idx:]
                mx   = np.max(np.abs(y))
                y_n  = y / mx if mx > 0 else y
                ax.plot(t_crop, y_n, linewidth=1.2)
                ax.set_title(COMPONENT_NAMES[comp], fontsize=9)
                ax.set_xlabel("Time (h)", fontsize=8)
                ax.set_ylabel("Norm. expression", fontsize=8)
                ax.tick_params(labelsize=7)
            plt.tight_layout()
            fig.savefig(
                os.path.join(param_dir, f"expr_plot_{param}_{val:.6f}.png"),
                dpi=120)
            plt.close(fig)

            # --- Phase portraits ---
            pairs    = list(combinations(COMPONENTS, 2))
            ncols    = 3
            nrows    = int(np.ceil(len(pairs) / ncols))
            fig2, axes2 = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
            axes2 = np.array(axes2).flatten()
            fig2.suptitle(f"Phase portraits — {param} = {val:.4g}", fontsize=11)
            for k, (c1, c2) in enumerate(pairs):
                axes2[k].plot(C[c1][burn_idx:], C[c2][burn_idx:],
                              linewidth=0.8, alpha=0.85)
                axes2[k].set_xlabel(COMPONENT_NAMES[c1], fontsize=8)
                axes2[k].set_ylabel(COMPONENT_NAMES[c2], fontsize=8)
                axes2[k].tick_params(labelsize=7)
            for k in range(len(pairs), len(axes2)):
                fig2.delaxes(axes2[k])
            plt.tight_layout()
            fig2.savefig(
                os.path.join(param_dir, f"phase_{param}_{val:.6f}.png"),
                dpi=120)
            plt.close(fig2)

            # --- Record loss of periodicity ---
            if all(np.isnan(p) or p == 0 for p in periods):
                if not loss_recorded:
                    with open(os.path.join(param_dir, "loss_threshold.txt"), "w") as f:
                        f.write(f"Periodicity lost at {param} = {val:.6f}\n")
                    loss_recorded = True

        print(f"  ✓ Done: {param}")
    except Exception as e:
        with open(os.path.join(results_dir, f"error_{param}.log"), "w") as f:
            f.write(str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mp.set_start_method('spawn', force=True)

    params_all  = define_parameters()
    skip        = {"eta1", "eta2"}
    param_items = [(k, v) for k, v in params_all.items() if k not in skip]

    results_dir = "parameter_analysis_ll"
    os.makedirs(results_dir, exist_ok=True)

    n_cores = min(8, mp.cpu_count())
    task_args = [(k, v, results_dir) for k, v in param_items]

    print(f"M1 LL parameter-range scan — {len(task_args)} parameters, {n_cores} workers")

    with mp.Pool(processes=n_cores) as pool:
        list(tqdm(pool.imap_unordered(analyze_parameter, task_args),
                  total=len(task_args),
                  desc="Parameters"))

    print(f"\nDone. Results in: {results_dir}/")


if __name__ == "__main__":
    main()
