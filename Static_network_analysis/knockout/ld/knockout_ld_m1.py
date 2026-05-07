"""
knockout_ld_m1.py
-----------------
Knockout analysis for the M1 circadian clock model under 12L:12D (LD 12:12)
conditions.  Parameters and equations are identical to knockout_ll_m1.py /
m1.py.  The only difference is the light-input protocol: a triangular ramp
light profile (matching m1.py) is applied over 10 entrainment days before the
analysis window, and simulations continue for a further 10 days from which
the final 48 h are used for period estimation.

Light protocol (matching m1.py)
--------------------------------
  LH = 6 h ramp-up  +  6 h ramp-down  (12 h light phase, peak = 26.62)
  DH = 12 h dark
  Pattern tiled over Nday = 20 days (480 h total).
  First 240 h (10 days) discarded as transient.

Photoreceptor flags per hour
-----------------------------
  Ired  > 0  and/or  Iblue > 0  →  ThetaPhyA = 1, else 0
  Ired  > 0                     →  ThetaPhyB = 1, else 0
  Iblue > 0                     →  ThetaCry1 = 1, else 0
"""

import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks
import pandas as pd
import os
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
OUT_DIR      = "knockout_results_ld"
PROFILES_DIR = os.path.join(OUT_DIR, "expression_profiles")
PLOTS_DIR    = os.path.join(OUT_DIR, "plots")
for d in [OUT_DIR, PROFILES_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters — identical to m1.py / knockout_ll_m1.py
# ---------------------------------------------------------------------------
def define_parameters():
    return {
        # CL (LHY/CCA1) module
        "v1":    4.8318,
        "q1a":   1.4266,
        "q3a":   8.9432,
        "q4a":   5.9277,
        "K1":    0.1943,
        "K2":    1.6138,
        "k1L":   0.2866,
        "k1D":   0.213,
        "p1":    0.8672,
        "p1L":   0.2378,
        "d1":    0.7843,
        # P97 (PRR9/PRR7) module
        "q1b":   3.575,
        "q3b":   5.5899,
        "q4b":   8.954,
        "v2":    1.6822,
        "K3":    2.2275,
        "K4":    0.40,
        "K5":    0.37,
        "k2":    0.35,
        "p2":    0.7858,
        "d2D":   0.3712,
        "d2L":   0.2917,
        # P51 (PRR5/TOC1) module
        "v3":    1.113,
        "K6":    0.4944,
        "K7":    2.4087,
        "k3":    0.5819,
        "p3":    0.6142,
        "d3D":   0.5026,
        "d3L":   0.5431,
        # EL (ELF4/LUX) module
        "v4":    2.5012,
        "K8":    0.3262,
        "K9":    1.7974,
        "K10":   1.1889,
        "k4":    0.925,
        "p4":    1.126,
        "de1":   0.0022,
        "de2":   0.4741,
        "de3":   0.3765,
        "de4":   0.398,
        "de5":   0.0003,
        # PhyA module
        "Ap3":   0.3868,
        "Am7":   0.5503,
        "Ak7":   1.125,
        "q2":    0.5767,
        "kmpac": 137,
        "kd":    7,
        # PIF module
        "v5":    0.1129,
        "K11":   0.3322,
        "K14":   1.5,
        "k5":    0.1591,
        "p5":    0.5293,
        "d5D":   0.4404,
        "d5L":   5.0712,
        # Hypocotyl
        "g1":    0.001,
        "g2":    0.18,
        "K12":   0.86,
        # PhyB module
        "Bp4":   0.4147,
        "Bm8":   0.7728,
        "Bk8":   0.1732,
        "kmpbc": 7162,
        # Cry1 module
        "Cp5":   0.4567,
        "Cm9":   0.867,
        "Ck9":   0.3237,
        "kmcc":  13406,
        # GZ module (direct protein dynamics, no separate mRNA state)
        "Gp6":   0.000100,
        "dg1":   0.010000,
        "dg2":   1.280202,
        "dg3":   0.010000,
        "dg4":   1.750462,
        "dg5":   1.067661,
        "dp6":   0.010000,
        "Gkp":   1.185527,
        # Light normalisation constants (fixed, not knocked out)
        "eta1":  0.03,
        "eta2":  0.0215,
    }

# ---------------------------------------------------------------------------
# Light schedule — triangular ramp, 12L:12D (matching m1.py exactly)
# ---------------------------------------------------------------------------
def build_light_schedule(Nday=20):
    """
    Returns (IntensityRR, IntensityBB) arrays of length Nday*24.

    DH = 12 dark hours
    LH = 6 ramp-up + 6 ramp-down  (peak intensity 26.62 µmol m⁻² s⁻¹)
    """
    DH = 12
    LH = (24 - DH) // 2          # 6 h each side of peak
    I_peak = 26.62

    one_day = np.concatenate((
        np.linspace(1, I_peak, LH),   # ramp up
        np.linspace(I_peak, 1, LH),   # ramp down
        np.zeros(DH),                  # dark
    ))                                 # length = 24

    schedule = np.tile(one_day, Nday)
    return schedule, schedule.copy()   # red == blue


# ---------------------------------------------------------------------------
# ODE right-hand side (identical equations to m1.py / knockout_ll_m1.py)
# ---------------------------------------------------------------------------
def model(p, Ired_arr, Iblue_arr):
    """
    Returns a closure rhs(C, t) compatible with scipy.integrate.odeint.
    Light intensities are looked up from pre-built hourly arrays using
    floor(t) as the index.
    """
    eta1 = p["eta1"]
    eta2 = p["eta2"]

    def rhs(C, t):
        idx   = int(t) % len(Ired_arr)
        Ired  = Ired_arr[idx]
        Iblue = Iblue_arr[idx]

        ThetaPhyA = 1.0 if (Ired > 0 or Iblue > 0) else 0.0
        ThetaPhyB = 1.0 if Ired  > 0 else 0.0
        ThetaCry1 = 1.0 if Iblue > 0 else 0.0

        from math import log 

        cop1_sum = C[14] + C[15] + C[16] + C[17]
        if cop1_sum == 0:
            cop1_sum = 1e-12

        dC = np.zeros(19)

        # Eq 0 — LHY mRNA
        dC[0] = (
            p["v1"]
            + (  p["q1a"] * (C[8]) * ThetaPhyA
               + p["q3a"] * (C[12]) * log(eta1 * Ired  + 1) * ThetaPhyB
               + p["q4a"] * (C[13]) * log(eta2 * Iblue + 1) * ThetaCry1)
        ) / (1 + (C[3] / p["K1"])**2 + (C[5] / p["K2"])**2) \
          - (p["k1L"] * ThetaPhyA + p["k1D"] * (1 - ThetaPhyA)) * C[0]

        # Eq 1 — LHY protein
        dC[1] = (p["p1"] + p["p1L"] * ThetaPhyA) * C[0] - p["d1"] * C[1]

        # Eq 2 — P97 mRNA
        dC[2] = (
            (  p["q1b"] * (C[8]) * ThetaPhyA
             + p["q3b"] * (C[12]) * log(eta1 * Ired  + 1) * ThetaPhyB
             + p["q4b"] * (C[13]) * log(eta2 * Iblue + 1) * ThetaCry1)
            + p["v2"]
        ) * (1 / (1 + (C[1] / p["K3"])**2 + (C[5] / p["K4"])**2 + (C[7] / p["K5"])**2)) \
          - p["k2"] * C[2]

        # Eq 3 — P97 protein
        dC[3] = p["p2"] * C[2] \
                - p["d2D"] * (1 - ThetaPhyA) * C[3] \
                - p["d2L"] * ThetaPhyA        * C[3]

        # Eq 4 — P51 mRNA  (K6, K7 only)
        dC[4] = p["v3"] / (1 + (C[1] / p["K6"])**2 + (C[5] / p["K7"])**2) \
                - p["k3"] * C[4]

        # Eq 5 — P51 protein  (includes GZ-mediated degradation)
        dC[5] = p["p3"] * C[4] \
                - p["d3D"] * (1 - ThetaPhyA) * C[5] \
                - p["d3L"] * ThetaPhyA        * C[5] \
                - (p["dp6"] * C[18] * C[5]) / (p["Gkp"] + C[5])

        # Eq 6 — EL mRNA
        dC[6] = p["v4"] * ThetaPhyA \
                / (1 + (C[1] / p["K8"])**2 + (C[5] / p["K9"])**2 + (C[7] / p["K10"])**2) \
                - p["k4"] * C[6]

        # Eq 7 — EL protein
        dC[7] = p["p4"] * C[6] \
                - (p["de1"]
                   + (p["de2"] * C[14] + p["de3"] * C[15]
                      + p["de4"] * C[16] + p["de5"] * C[17]) / cop1_sum) * C[7]

        # Eq 8 — PhyA
        dC[8] = (1 - ThetaPhyA) * p["Ap3"] \
                - (p["Am7"] * C[8] / (p["Ak7"] + C[8])) \
                - p["q2"]    * ThetaPhyA * C[8] \
                - p["kmpac"] * ThetaPhyA * C[8] * C[14] \
                + p["kd"] * C[15]

        # Eq 9 — PIF mRNA
        dC[9] = p["v5"] / (1 + (C[7] / p["K11"])**2 + (C[13] / p["K14"])**2) \
                - p["k5"] * C[9]

        # Eq 10 — PIF protein
        dC[10] = p["p5"] * C[9] \
                 - p["d5D"] * (1 - ThetaPhyA) * C[10] \
                 - p["d5L"] * ThetaPhyA        * C[10]

        # Eq 11 — HYP protein
        dC[11] = p["g1"] + (p["g2"] * C[10]**2) / (p["K12"]**2 + C[10]**2)

        # Eq 12 — PhyB
        dC[12] = p["Bp4"] \
                 - (p["Bm8"] * C[12] / (p["Bk8"] + C[12])) \
                 - p["kmpbc"] * ThetaPhyB * C[12] * C[14] \
                 + p["kd"] * C[16]

        # Eq 13 — Cry1
        dC[13] = p["Cp5"] \
                 - (p["Cm9"] * C[13] / (p["Ck9"] + C[13])) \
                 - p["kmcc"] * ThetaCry1 * C[13] * C[14] \
                 + p["kd"] * C[17]

        # Eq 14 — COP1
        dC[14] = (
            - p["kmpac"]  * ThetaPhyA * C[8]  * C[14] + p["kd"] * C[15]
            - p["kmpbc"]  * ThetaPhyB * C[12] * C[14] + p["kd"] * C[16]
            - p["kmcc"]   * ThetaCry1 * C[13] * C[14] + p["kd"] * C[17]
            + (p["Am7"] * C[15] / (p["Ak7"] + C[15]))
            +  p["q2"]   * ThetaPhyA * C[15]
            + (p["Bm8"] * C[16] / (p["Bk8"] + C[16]))
            + (p["Cm9"] * C[17] / (p["Ck9"] + C[17]))
        )

        # Eq 15 — COP1:PhyA
        dC[15] = p["kmpac"] * ThetaPhyA * C[8]  * C[14] \
                 - p["kd"] * C[15] \
                 - (p["Am7"] * C[15] / (p["Ak7"] + C[15])) \
                 - p["q2"] * ThetaPhyA * C[15]

        # Eq 16 — COP1:PhyB
        dC[16] = p["kmpbc"] * ThetaPhyB * C[12] * C[14] \
                 - p["kd"] * C[16] \
                 - (p["Bm8"] * C[16] / (p["Bk8"] + C[16]))

        # Eq 17 — COP1:Cry1
        dC[17] = p["kmcc"] * ThetaCry1 * C[13] * C[14] \
                 - p["kd"] * C[17] \
                 - (p["Cm9"] * C[17] / (p["Ck9"] + C[17]))

        # Eq 18 — GZ protein
        dC[18] = p["Gp6"] \
                 - (p["dg1"]
                    + (p["dg2"] * C[14] + p["dg3"] * C[15]
                       + p["dg4"] * C[16] + p["dg5"] * C[17]) / cop1_sum) * C[18]

        return dC

    return rhs


# ---------------------------------------------------------------------------
# Simulation — LD 12:12, hourly stepping (matching m1.py integration loop)
# ---------------------------------------------------------------------------
def run_simulation(params, Nday=20):
    """
    Integrate the M1 model under LD 12:12 using the same hour-by-hour
    odeint loop as m1.py.

    Returns
    -------
    time : 1-D array, length Nday*24 + 1  (hours 0 … Nday*24)
    C    : 2-D array, shape (19, Nday*24 + 1)
    """
    IntensityRR, IntensityBB = build_light_schedule(Nday)
    rhs = model(params, IntensityRR, IntensityBB)

    C0 = np.ones(19)
    C0[11] = C0[15] = C0[16] = C0[17] = 0.0

    C_all = [C0.copy()]
    for t in range(len(IntensityRR)):
        tspan = [t, t + 1]
        sol   = odeint(rhs, C_all[-1], tspan)
        C_all.append(sol[-1])

    C_arr = np.array(C_all).T          # shape (19, Nday*24 + 1)
    time  = np.arange(C_arr.shape[1], dtype=float)
    return time, C_arr


# ---------------------------------------------------------------------------
# Period estimation (same logic as knockout_ll_m1.py)
# ---------------------------------------------------------------------------
def compute_period(time, signal, burn_in=240):
    """
    Estimate mean period from peaks after discarding the first burn_in hours.
    Returns np.nan if arrhythmic or too few peaks.
    """
    idx   = np.searchsorted(time, burn_in)
    t     = time[idx:]
    s     = signal[idx:]

    s_range = np.max(s) - np.min(s)
    if s_range < 0.1 * np.max(np.abs(s)) + 1e-12:
        return np.nan

    s_norm = (s - np.min(s)) / (s_range + 1e-12)
    peaks, _ = find_peaks(s_norm, prominence=0.05)
    if len(peaks) < 3:
        return np.nan

    return float(np.mean(np.diff(t[peaks])))


def normalize_signal(signal):
    mx = np.max(np.abs(signal))
    return signal / mx if mx > 0 else np.zeros_like(signal)


# ---------------------------------------------------------------------------
# Worker (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------
def _run_one_param(args):
    param, base_params = args
    rows  = []
    cache = {}

    for value, label in [(base_params[param], "baseline"), (1e-9, "knockout")]:
        p = dict(base_params)
        p[param] = value

        try:
            time, C = run_simulation(p)
        except Exception as e:
            print(f"  [warn] {param}={value}: solver failed — {e}")
            n    = 481
            time = np.arange(n, dtype=float)
            C    = np.zeros((19, n))

        periods = [compute_period(time, C[i]) for i in [0, 2, 4, 6]]

        burn_idx = np.searchsorted(time, 240)
        t_crop   = time[burn_idx:] - time[burn_idx]
        raw = {
            "LHYm": C[0][burn_idx:],
            "P97m": C[2][burn_idx:],
            "P51m": C[4][burn_idx:],
            "ELm":  C[6][burn_idx:],
        }
        norm = {k: normalize_signal(v) for k, v in raw.items()}
        cache[label] = (value, t_crop, raw, norm)

        rows.append({
            "param":       param,
            "value":       value,
            "label":       label,
            "Period_LHYm": periods[0],
            "Period_P97m": periods[1],
            "Period_P51m": periods[2],
            "Period_ELm":  periods[3],
        })

    return param, rows, cache


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def _save_comparison_plot(param, cache):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    species  = [("LHYm", (0, 0)), ("P97m", (0, 1)),
                ("P51m", (1, 0)), ("ELm",  (1, 1))]
    colours  = {"baseline": "#2C72B5", "knockout": "#D94C2B"}

    for name, (row, col) in species:
        ax = axes[row][col]
        for lbl in ["baseline", "knockout"]:
            val, t, _, norm = cache[lbl]
            ax.plot(t, norm[name],
                    label=f"{lbl}  ({param}={val:.3g})",
                    color=colours[lbl], linewidth=1.2)

        # shade dark phases (12 h light, 12 h dark; first dark starts at h 12)
        t_max = t[-1] if len(t) > 0 else 240
        dark_start = 12
        while dark_start < t_max:
            ax.axvspan(dark_start, min(dark_start + 12, t_max),
                       facecolor="grey", alpha=0.15, linewidth=0)
            dark_start += 24

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.set_ylabel("Norm. expression", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"LD 12:12 knockout — {param}", y=0.98)
    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center",
               bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(PLOTS_DIR, f"{param}_norm_profiles_ld.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_params = define_parameters()
    skip        = {"eta1", "eta2"}
    param_names = [k for k in base_params if k not in skip]

    outfile = os.path.join(OUT_DIR, "knockout_analysis_ld.csv")
    with open(outfile, "w") as f:
        f.write("Parameter,Value,Label,Period_LHYm,Period_P97m,Period_P51m,Period_ELm\n")

    n_cores   = min(8, mp.cpu_count())
    task_args = [(p, base_params) for p in param_names]

    print(f"M1 LD 12:12 knockout — {len(param_names)} parameters, {n_cores} workers")

    all_results = []

    with mp.Pool(processes=n_cores) as pool:
        for param, rows, cache in tqdm(
                pool.imap_unordered(_run_one_param, task_args),
                total=len(task_args),
                desc="Knockout Analysis (LD 12:12)"):

            all_results.extend(rows)

            for row in rows:
                lbl = row["label"]
                val, t_crop, raw, norm = cache[lbl]
                df = pd.DataFrame({
                    "Time":      t_crop,
                    "LHYm_raw":  raw["LHYm"],
                    "P97m_raw":  raw["P97m"],
                    "P51m_raw":  raw["P51m"],
                    "ELm_raw":   raw["ELm"],
                    "LHYm_norm": norm["LHYm"],
                    "P97m_norm": norm["P97m"],
                    "P51m_norm": norm["P51m"],
                    "ELm_norm":  norm["ELm"],
                })
                df.to_csv(os.path.join(PROFILES_DIR,
                                       f"{param}_{lbl}_ld.csv"), index=False)

                with open(outfile, "a") as f:
                    f.write(f"{param},{val},{lbl},"
                            f"{row['Period_LHYm']},{row['Period_P97m']},"
                            f"{row['Period_P51m']},{row['Period_ELm']}\n")

            _save_comparison_plot(param, cache)

    # Consolidated summary
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(OUT_DIR, "knockout_summary_ld.csv"), index=False)

    # Δperiod table (knockout − baseline)
    period_cols = ["Period_LHYm", "Period_P97m", "Period_P51m", "Period_ELm"]
    df_base  = df_all[df_all["label"] == "baseline"].set_index("param")[period_cols]
    df_ko    = df_all[df_all["label"] == "knockout"].set_index("param")[period_cols]
    df_delta = (df_ko - df_base).rename(columns={c: "Delta_" + c for c in period_cols})
    df_delta.to_csv(os.path.join(OUT_DIR, "knockout_delta_period_ld.csv"))

    print(f"\nDone.  All results saved to: {OUT_DIR}/")
    print(f"  knockout_analysis_ld.csv     — per-run periods")
    print(f"  knockout_summary_ld.csv      — full summary table")
    print(f"  knockout_delta_period_ld.csv — Δperiod (KO − baseline)")
    print(f"  {PROFILES_DIR}/             — expression profile CSVs")
    print(f"  {PLOTS_DIR}/                — comparison plots")


if __name__ == "__main__":
    main()
