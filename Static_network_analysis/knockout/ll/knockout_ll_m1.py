"""
knockout_ll_m1.py
-----------------
Knockout analysis for the M1 circadian clock model under constant-light (LL)
conditions.  Parameters, equations, and state-vector layout are taken directly
from m1.py (19 ODEs, GZ module without a separate mRNA state).

Key differences from the original knockout_ll.py
-------------------------------------------------
* 19-state ODE system (C[0]–C[18]) matching m1.py exactly.
* GZ parameters replaced: Gp6, dg1-dg5, dp6, Gkp  (removed v6,k6,K15,p6,d6D,d6L).
* P51 protein (Eq 5) includes the GZ-mediated degradation term  dp6·C[18]·C[5]/(Gkp+C[5]).
* CL/P97 mRNA light terms use C[8]+C[15] and C[12]+C[16] instead of C[8],C[12].
* P51 mRNA (Eq 4) has only K6, K7 inhibitors (K13 term removed).
* LL condition: ThetaPhyA=ThetaPhyB=ThetaCry1=1, Ired=Iblue=26.62 (constant).
* Integration: same odeint hour-by-hour loop as m1.py (Nday=20, 480 h total).
* First 240 h discarded as transient.
* Parallel execution via multiprocessing (8 cores by default).
* Period computed from mRNA oscillations of LHYm(0), P97m(2), P51m(4), ELm(6).
"""

import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks
from math import log
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
OUT_DIR       = "knockout_results_ll"
PROFILES_DIR  = os.path.join(OUT_DIR, "expression_profiles")
PLOTS_DIR     = os.path.join(OUT_DIR, "plots")
for d in [OUT_DIR, PROFILES_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters — taken verbatim from m1.py
# ---------------------------------------------------------------------------
def define_parameters():
    return {
        # CL (LHY/CCA1) module
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
        # P97 (PRR9/PRR7) module
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
        # P51 (PRR5/TOC1) module
        "v3":    1.113,    # P51 basal synthesis
        "K6":    0.4944,   # Inhibition of P51 by CL
        "K7":    2.4087,   # Inhibition of P51 by P51
        "k3":    0.5819,   # P51 mRNA degradation
        "p3":    0.6142,   # P51 translation
        "d3D":   0.5026,   # P51 protein degradation (dark)
        "d3L":   0.5431,   # P51 protein degradation (light)
        # EL (ELF4/LUX) module
        "v4":    2.5012,   # EL basal synthesis
        "K8":    0.3262,   # Inhibition of EL by CL
        "K9":    1.7974,   # Inhibition of EL by P51
        "K10":   1.1889,   # Inhibition of EL by EL protein
        "k4":    0.925,    # EL mRNA degradation
        "p4":    1.126,    # EL translation
        "de1":   0.0022,   # EL basal degradation
        "de2":   0.4741,   # EL degradation (free COP1)
        "de3":   0.3765,   # EL degradation (COP1:PhyA)
        "de4":   0.398,    # EL degradation (COP1:PhyB)
        "de5":   0.0003,   # EL degradation (COP1:Cry1)
        # PhyA module
        "Ap3":   0.3868,   # PhyA synthesis
        "Am7":   0.5503,   # PhyA Michaelis degradation Vmax
        "Ak7":   1.125,    # PhyA Michaelis constant
        "q2":    0.5767,   # Light-independent PhyA inactivation
        "kmpac": 137,      # COP1:PhyA binding rate
        "kd":    7,        # COP1 complex dissociation rate
        # PIF module
        "v5":    0.1129,   # PIF basal synthesis
        "K11":   0.3322,   # Inhibition of PIF by EL
        "K14":   1.5,      # Inhibition of PIF by Cry1
        "k5":    0.1591,   # PIF mRNA degradation
        "p5":    0.5293,   # PIF translation
        "d5D":   0.4404,   # PIF protein degradation (dark)
        "d5L":   5.0712,   # PIF protein degradation (light)
        # Hypocotyl
        "g1":    0.001,    # Basal hypocotyl growth
        "g2":    0.18,     # PIF-induced hypocotyl growth
        "K12":   0.86,     # Activation of growth by PIF
        # PhyB module
        "Bp4":   0.4147,   # PhyB synthesis
        "Bm8":   0.7728,   # PhyB Michaelis degradation Vmax
        "Bk8":   0.1732,   # PhyB Michaelis constant
        "kmpbc": 7162,     # COP1:PhyB binding rate
        # Cry1 module
        "Cp5":   0.4567,   # Cry1 synthesis
        "Cm9":   0.867,    # Cry1 Michaelis degradation Vmax
        "Ck9":   0.3237,   # Cry1 Michaelis constant
        "kmcc":  13406,    # COP1:Cry1 binding rate
        # GZ module (m1.py form — direct protein dynamics, no mRNA state)
        "Gp6":   0.000100, # GZ synthesis rate
        "dg1":   0.010000, # GZ basal degradation
        "dg2":   1.280202, # GZ degradation (free COP1)
        "dg3":   0.010000, # GZ degradation (COP1:PhyA)
        "dg4":   1.750462, # GZ degradation (COP1:PhyB)
        "dg5":   1.067661, # GZ degradation (COP1:Cry1)
        "dp6":   0.010000, # GZ-mediated P51 protein degradation rate
        "Gkp":   1.185527, # Michaelis constant for GZ-mediated P51 degradation
        # Light normalisation constants (fixed, not knocked out)
        "eta1":  0.03,
        "eta2":  0.0215,
    }

# ---------------------------------------------------------------------------
# ODE system — 19 states, structure identical to m1.py
# ---------------------------------------------------------------------------
def model(t, C, p,
              ThetaPhyA=1.0, ThetaPhyB=1.0, ThetaCry1=1.0,
              Ired=26.62,    Iblue=26.62):
    """
    Right-hand side of the M1 ODE system.

    State vector (19 elements)
    --------------------------
    C[0]  LHY mRNA      C[1]  LHY protein
    C[2]  P97 mRNA      C[3]  P97 protein
    C[4]  P51 mRNA      C[5]  P51 protein
    C[6]  EL  mRNA      C[7]  EL  protein
    C[8]  PhyA          C[9]  PIF mRNA
    C[10] PIF protein   C[11] HYP protein
    C[12] PhyB          C[13] Cry1
    C[14] COP1          C[15] COP1:PhyA
    C[16] COP1:PhyB     C[17] COP1:Cry1
    C[18] GZ protein
    """
    # Guard against zero denominator in weighted COP1 degradation terms
    cop1_sum = C[14] + C[15] + C[16] + C[17]
    if cop1_sum == 0:
        cop1_sum = 1e-12

    dC = np.zeros(19)

    # --- Eq 0  LHY mRNA ---------------------------------------------------
    dC[0] = (
        p["v1"]
        + (  p["q1a"] * (C[8]) * ThetaPhyA
           + p["q3a"] * (C[12]) * log(p["eta1"] * Ired  + 1) * ThetaPhyB
           + p["q4a"] * (C[13]) * log(p["eta2"] * Iblue + 1) * ThetaCry1)
    ) / (1 + (C[3] / p["K1"])**2 + (C[5] / p["K2"])**2) \
      - (p["k1L"] * ThetaPhyA + p["k1D"] * (1 - ThetaPhyA)) * C[0]

    # --- Eq 1  LHY protein ------------------------------------------------
    dC[1] = (p["p1"] + p["p1L"] * ThetaPhyA) * C[0] - p["d1"] * C[1]

    # --- Eq 2  P97 mRNA ---------------------------------------------------
    dC[2] = (
        (  p["q1b"] * (C[8]) * ThetaPhyA
         + p["q3b"] * (C[12]) * log(p["eta1"] * Ired  + 1) * ThetaPhyB
         + p["q4b"] * (C[13]) * log(p["eta2"] * Iblue + 1) * ThetaCry1)
        + p["v2"]
    ) * (1 / (1 + (C[1] / p["K3"])**2 + (C[5] / p["K4"])**2 + (C[7] / p["K5"])**2)) \
      - p["k2"] * C[2]

    # --- Eq 3  P97 protein ------------------------------------------------
    dC[3] = p["p2"] * C[2] \
            - p["d2D"] * (1 - ThetaPhyA) * C[3] \
            - p["d2L"] * ThetaPhyA        * C[3]

    # --- Eq 4  P51 mRNA  (K6, K7 only — as in m1.py; no K13 term) --------
    dC[4] = p["v3"] / (1 + (C[1] / p["K6"])**2 + (C[5] / p["K7"])**2) \
            - p["k3"] * C[4]

    # --- Eq 5  P51 protein  (includes GZ-mediated Michaelis degradation) --
    dC[5] = p["p3"] * C[4] \
            - p["d3D"] * (1 - ThetaPhyA) * C[5] \
            - p["d3L"] * ThetaPhyA        * C[5] \
            - (p["dp6"] * C[18] * C[5]) / (p["Gkp"] + C[5])

    # --- Eq 6  EL mRNA ----------------------------------------------------
    dC[6] = p["v4"] * ThetaPhyA \
            / (1 + (C[1] / p["K8"])**2 + (C[5] / p["K9"])**2 + (C[7] / p["K10"])**2) \
            - p["k4"] * C[6]

    # --- Eq 7  EL protein -------------------------------------------------
    dC[7] = p["p4"] * C[6] \
            - (p["de1"]
               + (p["de2"] * C[14] + p["de3"] * C[15]
                  + p["de4"] * C[16] + p["de5"] * C[17]) / cop1_sum) * C[7]

    # --- Eq 8  PhyA -------------------------------------------------------
    dC[8] = (1 - ThetaPhyA) * p["Ap3"] \
            - (p["Am7"] * C[8] / (p["Ak7"] + C[8])) \
            - p["q2"]    * ThetaPhyA * C[8] \
            - p["kmpac"] * ThetaPhyA * C[8] * C[14] \
            + p["kd"] * C[15]

    # --- Eq 9  PIF mRNA ---------------------------------------------------
    dC[9] = p["v5"] / (1 + (C[7] / p["K11"])**2 + (C[13] / p["K14"])**2) \
            - p["k5"] * C[9]

    # --- Eq 10 PIF protein ------------------------------------------------
    dC[10] = p["p5"] * C[9] \
             - p["d5D"] * (1 - ThetaPhyA) * C[10] \
             - p["d5L"] * ThetaPhyA        * C[10]

    # --- Eq 11 HYP protein ------------------------------------------------
    dC[11] = p["g1"] + (p["g2"] * C[10]**2) / (p["K12"]**2 + C[10]**2)

    # --- Eq 12 PhyB -------------------------------------------------------
    dC[12] = p["Bp4"] \
             - (p["Bm8"] * C[12] / (p["Bk8"] + C[12])) \
             - p["kmpbc"] * ThetaPhyB * C[12] * C[14] \
             + p["kd"] * C[16]

    # --- Eq 13 Cry1 -------------------------------------------------------
    dC[13] = p["Cp5"] \
             - (p["Cm9"] * C[13] / (p["Ck9"] + C[13])) \
             - p["kmcc"] * ThetaCry1 * C[13] * C[14] \
             + p["kd"] * C[17]

    # --- Eq 14 COP1 -------------------------------------------------------
    dC[14] = (
        - p["kmpac"]  * ThetaPhyA  * C[8]  * C[14]  + p["kd"] * C[15]
        - p["kmpbc"]  * ThetaPhyB  * C[12] * C[14]  + p["kd"] * C[16]
        - p["kmcc"]   * ThetaCry1  * C[13] * C[14]  + p["kd"] * C[17]
        + (p["Am7"] * C[15] / (p["Ak7"] + C[15]))
        +  p["q2"]   * ThetaPhyA  * C[15]
        + (p["Bm8"] * C[16] / (p["Bk8"] + C[16]))
        + (p["Cm9"] * C[17] / (p["Ck9"] + C[17]))
    )

    # --- Eq 15 COP1:PhyA --------------------------------------------------
    dC[15] = p["kmpac"] * ThetaPhyA * C[8]  * C[14] \
             - p["kd"] * C[15] \
             - (p["Am7"] * C[15] / (p["Ak7"] + C[15])) \
             - p["q2"] * ThetaPhyA * C[15]

    # --- Eq 16 COP1:PhyB --------------------------------------------------
    dC[16] = p["kmpbc"] * ThetaPhyB * C[12] * C[14] \
             - p["kd"] * C[16] \
             - (p["Bm8"] * C[16] / (p["Bk8"] + C[16]))

    # --- Eq 17 COP1:Cry1 --------------------------------------------------
    dC[17] = p["kmcc"] * ThetaCry1 * C[13] * C[14] \
             - p["kd"] * C[17] \
             - (p["Cm9"] * C[17] / (p["Ck9"] + C[17]))

    # --- Eq 18 GZ protein  (direct synthesis/degradation, no mRNA state) --
    dC[18] = p["Gp6"] \
             - (p["dg1"]
                + (p["dg2"] * C[14] + p["dg3"] * C[15]
                   + p["dg4"] * C[16] + p["dg5"] * C[17]) / cop1_sum) * C[18]

    return dC


# ---------------------------------------------------------------------------
# Simulation (LL: constant light throughout, odeint hourly loop)
# ---------------------------------------------------------------------------
def run_simulation(params, Nday=20):
    """
    Run the M1 model under constant-light (LL) using the same hour-by-hour
    odeint loop as m1.py.  Ired = Iblue = 26.62 every hour.

    Returns (time, C_array) where C_array has shape (19, Nday*24 + 1).
    """
    ThetaPhyA = ThetaPhyB = ThetaCry1 = 1.0
    Ired = Iblue = 26.62

    def rhs(C, t):
        return model(t, C, params,
                         ThetaPhyA=ThetaPhyA, ThetaPhyB=ThetaPhyB,
                         ThetaCry1=ThetaCry1, Ired=Ired, Iblue=Iblue)

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
# Period estimation
# ---------------------------------------------------------------------------
def compute_period(time, signal, burn_in=240):
    """
    Estimate the mean period of a signal after discarding the first
    `burn_in` hours as transient.  Returns np.nan if the signal is
    arrhythmic or has insufficient peaks.
    """
    idx = np.searchsorted(time, burn_in)
    t   = time[idx:]
    s   = signal[idx:]

    s_range = np.max(s) - np.min(s)
    if s_range < 0.1 * np.max(np.abs(s)) + 1e-12:
        return np.nan    # essentially flat — arrhythmic

    s_norm = (s - np.min(s)) / (s_range + 1e-12)
    peaks, _ = find_peaks(s_norm, prominence=0.05)
    if len(peaks) < 3:
        return np.nan

    return float(np.mean(np.diff(t[peaks])))


def normalize_signal(signal):
    mx = np.max(np.abs(signal))
    return signal / mx if mx > 0 else np.zeros_like(signal)


# ---------------------------------------------------------------------------
# Worker function (used in parallel pool)
# ---------------------------------------------------------------------------
def _run_one_param(args):
    """
    Run baseline and knockout for a single parameter.
    Returns (param, rows, cache).
    Must be a top-level function for multiprocessing pickling.
    """
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
            n = 481
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
    species = [("LHYm", (0, 0)), ("P97m", (0, 1)),
               ("P51m", (1, 0)), ("ELm",  (1, 1))]
    colours = {"baseline": "#2C72B5", "knockout": "#D94C2B"}

    for name, (row, col) in species:
        ax = axes[row][col]
        for lbl in ["baseline", "knockout"]:
            val, t, _, norm = cache[lbl]
            ax.plot(t, norm[name],
                    label=f"{lbl}  ({param}={val:.3g})",
                    color=colours[lbl], linewidth=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.set_ylabel("Norm. expression", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"LL knockout — {param}", y=0.98)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(PLOTS_DIR, f"{param}_norm_profiles_ll.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_params = define_parameters()
    # Exclude pure normalisation constants (not biologically meaningful to knock out)
    skip = {"eta1", "eta2"}
    param_names = [k for k in base_params if k not in skip]

    outfile = os.path.join(OUT_DIR, "knockout_analysis_ll.csv")
    with open(outfile, "w") as f:
        f.write("Parameter,Value,Label,Period_LHYm,Period_P97m,Period_P51m,Period_ELm\n")

    n_cores = min(8, mp.cpu_count())
    task_args = [(p, base_params) for p in param_names]

    print(f"M1 LL knockout — {len(param_names)} parameters, {n_cores} workers")

    all_results = []

    with mp.Pool(processes=n_cores) as pool:
        for param, rows, cache in tqdm(
                pool.imap_unordered(_run_one_param, task_args),
                total=len(task_args),
                desc="Knockout Analysis"):

            all_results.extend(rows)

            # Save per-condition expression profiles to CSV
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
                                       f"{param}_{lbl}_ll.csv"), index=False)

                with open(outfile, "a") as f:
                    f.write(f"{param},{val},{lbl},"
                            f"{row['Period_LHYm']},{row['Period_P97m']},"
                            f"{row['Period_P51m']},{row['Period_ELm']}\n")

            _save_comparison_plot(param, cache)

    # Consolidated summary
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(OUT_DIR, "knockout_summary_ll.csv"), index=False)

    # Δperiod table (knockout − baseline) for quick inspection
    period_cols = ["Period_LHYm", "Period_P97m", "Period_P51m", "Period_ELm"]
    df_base  = df_all[df_all["label"] == "baseline"].set_index("param")[period_cols]
    df_ko    = df_all[df_all["label"] == "knockout"].set_index("param")[period_cols]
    df_delta = (df_ko - df_base).rename(columns={c: "Delta_" + c for c in period_cols})
    df_delta.to_csv(os.path.join(OUT_DIR, "knockout_delta_period_ll.csv"))

    print(f"\nDone.  All results saved to: {OUT_DIR}/")
    print(f"  knockout_analysis_ll.csv     — per-run periods")
    print(f"  knockout_summary_ll.csv      — full summary table")
    print(f"  knockout_delta_period_ll.csv — Δperiod (KO − baseline)")
    print(f"  {PROFILES_DIR}/             — expression profile CSVs")
    print(f"  {PLOTS_DIR}/                — comparison plots")


if __name__ == "__main__":
    main()
