# M1 Plant Circadian Network Architecture

A computational systems biology project modeling the *Arabidopsis thaliana* circadian clock using a 19-variable ODE system (M1 model). The pipeline covers model simulation, parameter sensitivity analysis, phase-space geometry, gene knockout effects, and network topology — all under both standard and natural seasonal light conditions.

---

## Table of Contents

1. [Background](#background)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Running the Analysis](#running-the-analysis)
   - [Step 1 — Core Model](#step-1--core-model)
   - [Step 2 — Parameter Sweeps](#step-2--parameter-sweeps)
   - [Step 3 — Period / Phase / Knockout Analysis](#step-3--period--phase--knockout-analysis)
   - [Step 4 — Comparative Analysis and Network Construction](#step-4--comparative-analysis-and-network-construction)
   - [Natural Light Validation](#natural-light-validation)
   - [Hypocotyl Growth Extension](#hypocotyl-growth-extension)
6. [Outputs](#outputs)
7. [Model Description](#model-description)
8. [Data Formats](#data-formats)
9. [Citation](#citation)

---

## Background

The circadian clock of *Arabidopsis thaliana* is driven by interlocking transcription–translation feedback loops involving four gene clusters:

| Symbol | Genes          |
|--------|----------------|
| CL     | LHY / CCA1     |
| P97    | PRR9 / PRR7    |
| P51    | PRR5 / TOC1    |
| EL     | ELF4 / LUX     |

The **M1 model** extends the legacy "Pay" model by incorporating:

- Phytochrome A (PhyA) and Phytochrome B (PhyB) red-light signalling
- Cryptochrome 1 (Cry1) blue-light signalling
- COP1-mediated protein degradation complexes
- PIF (Phytochrome Interacting Factor) and hypocotyl growth dynamics
- A GZ regulatory protein

The model is validated against experimental expression data under four natural seasonal photoperiods and against the Pay model benchmark.

---

## Project Structure

```
M1_Plant_Circadian_Network_Architecture/
│
├── model/
│   ├── 12l12d/                          # Standard 12 h light : 12 h dark cycle
│   │   ├── m1.py                        # M1 ODE model — main simulation
│   │   ├── pay_og.py                    # Legacy Pay model (benchmark)
│   │   ├── comparison_plots.py          # MAE/MSE bar and scatter plots
│   │   ├── model1_extracted_data.csv    # M1 mRNA time-series output
│   │   ├── pay_model_extracted_data.csv # Pay model mRNA time-series output
│   │   ├── model1_performance.txt       # M1 MAE and MSE per gene
│   │   └── pay_performance.txt          # Pay model MAE and MSE per gene
│   │
│   └── hypocotyl/
│       └── m1_with_hypo_plot.py         # M1 + PIF-driven hypocotyl growth
│
├── model_with_natural_data/             # Seasonal validation (4 photoperiods)
│   ├── trial1_natural_autumn/           # Autumn equinox  (22 Sep)
│   ├── trial2_natural_spring/           # Spring equinox  (20 Mar)
│   ├── trial3_natural_summer/           # Summer solstice (21 Jun)
│   └── trial4_natural_winter/           # Winter solstice (21 Dec)
│       (each contains m1.py, comparison_plots.py, .xlsx data, extracted_data.csv)
│
└── Static_network_analysis/            # Sensitivity, knockout, network analysis
    ├── period_range_auto_ll_m1.py       # Parameter sweep — LL (constant light)
    ├── period_range_auto_ld_m1.py       # Parameter sweep — LD (light/dark cycle)
    ├── analyse_and_plot.py              # General expression / phase-portrait plots
    │
    ├── Period/
    │   ├── period_analysis_pipeline_LL.py  # Period sensitivity — LL
    │   └── period_analysis_pipeline_LD.py  # Period sensitivity — LD
    │
    ├── Phase/
    │   ├── phase_pipeline_LL.py            # Phase geometry — LL
    │   └── phase_pipeline_LD.py            # Phase geometry — LD
    │
    ├── knockout/
    │   ├── ll/
    │   │   ├── knockout_ll_m1.py            # Run knockouts — LL
    │   │   └── knockout_full_pipeline_LL.py # Visualise knockout results — LL
    │   └── ld/
    │       ├── knockout_ld_m1.py            # Run knockouts — LD
    │       └── knockout_full_pipeline_LD.py # Visualise knockout results — LD
    │
    ├── comparative/
    │   ├── period/
    │   │   └── plot_period_change.py        # LL vs LD period fold-change plots
    │   ├── phase/
    │   │   ├── rate_plots.py                # LL vs LD phase geometry comparison
    │   │   └── export_conclusive_mean_area_eccentricity_excel.py
    │   └── knockout/
    │       └── plot_knockout_compare.py     # LL vs LD knockout dumbbell plots
    │
    └── network/
        └── build_weighted_networks.py       # Build Cytoscape-ready network files
```

---

## Requirements

- **Python** 3.7 or later

| Package         | Purpose                                    |
|-----------------|--------------------------------------------|
| `numpy`         | Numerical arrays                           |
| `scipy`         | ODE solver (`odeint`), signal processing   |
| `pandas`        | CSV/Excel I/O and DataFrames               |
| `matplotlib`    | Plotting (publication figures)             |
| `seaborn`       | Heatmaps and statistical plots             |
| `networkx`      | Network graph construction                 |
| `opencv-python` | Phase-portrait image processing            |
| `Pillow`        | Image I/O                                  |
| `openpyxl`      | Excel `.xlsx` file generation              |
| `tqdm`          | Progress bars for parameter sweeps         |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd M1_Plant_Circadian_Network_Architecture

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install numpy scipy pandas matplotlib seaborn networkx \
            opencv-python Pillow openpyxl tqdm
```

No additional configuration files are required — all model parameters are hard-coded in the respective `m1.py` scripts.

---

## Running the Analysis

Run each step from its own directory as shown, or supply the full path. Outputs are written relative to the script's location unless noted otherwise.

### Step 1 — Core Model

Simulate the M1 model under a standard 12 h : 12 h light/dark cycle and compare it to the Pay benchmark.

```bash
# Run M1 model (generates waveform PNGs and model1_extracted_data.csv)
python model/12l12d/m1.py

# Run legacy Pay model (generates pay_model_extracted_data.csv)
python model/12l12d/pay_og.py

# Generate MAE / MSE comparison bar and scatter plots
python model/12l12d/comparison_plots.py
```

Key outputs in `model/12l12d/`:
- `model1_extracted_data.csv` — hourly mRNA time-series (LHYm, P97m, P51m, ELm)
- `model1_performance.txt` — overall and per-gene MAE and MSE
- Waveform PNGs for all 19 state variables
- `MAE_comparison_bar.png`, `MAE_pay_vs_M1_scatter.png`

---

### Step 2 — Parameter Sweeps

Sweep all 116 model parameters individually (±20–30 % of base values, ~10 points each) under constant-light (LL) and light/dark (LD) conditions. Uses multiprocessing (up to 8 cores).

```bash
cd Static_network_analysis

# LL sweep  (~30–60 min depending on hardware)
python period_range_auto_ll_m1.py

# LD sweep
python period_range_auto_ld_m1.py
```

Outputs:
- `Static_network_analysis/parameter_analysis_ll/` — one sub-folder per parameter, each containing `expression_profile_<param>_<value>.csv`
- `Static_network_analysis/parameter_analysis_ld/` — same for LD

These folders are consumed by Steps 3 and 4.

---

### Step 3 — Period / Phase / Knockout Analysis

All scripts in this step read from the `parameter_analysis_ll/` or `parameter_analysis_ld/` folders produced in Step 2.

#### 3a — Period Sensitivity

```bash
python Static_network_analysis/Period/period_analysis_pipeline_LL.py
python Static_network_analysis/Period/period_analysis_pipeline_LD.py
```

Outputs in `LL_period_outputs/` / `LD_period_outputs/`:
- Period-vs-parameter line plots
- Sensitivity heatmaps
- `parameter_period_deltas_ll.csv`, `parameter_sensitivity_summary_ll.csv`

#### 3b — Phase-Space Geometry

```bash
python Static_network_analysis/Phase/phase_pipeline_LL.py
python Static_network_analysis/Phase/phase_pipeline_LD.py
```

Outputs in `LL_outputs/` / `LD_outputs/`:
- Phase-portrait PNGs (convex-hull area and eccentricity per parameter)
- `mean_metrics_ll.csv`, `mean_metrics_ld.csv`

#### 3c — Knockout Analysis

```bash
# Simulate knockouts and collect period data
python Static_network_analysis/knockout/ll/knockout_ll_m1.py
python Static_network_analysis/knockout/ld/knockout_ld_m1.py

# Generate visualisation plots
python Static_network_analysis/knockout/ll/knockout_full_pipeline_LL.py
python Static_network_analysis/knockout/ld/knockout_full_pipeline_LD.py
```

Outputs:
- `knockout_analysis_ll.csv` / `knockout_analysis_ld.csv` — period under baseline vs knockout per gene
- `knockout_output_LL/` / `knockout_output_LD/` — bar plots and waveform overlays
- Parameters classified into Class I (arrhythmic), Class II (moderate), Class III (minor effect)

---

### Step 4 — Comparative Analysis and Network Construction

Combine LL and LD results to identify condition-specific and shared regulatory features.

```bash
# Period fold-change comparison (LL vs LD overlay)
python Static_network_analysis/comparative/period/plot_period_change.py

# Phase geometry rate-of-change comparison
python Static_network_analysis/comparative/phase/rate_plots.py

# Export integrated phase metrics to Excel
python Static_network_analysis/comparative/phase/export_conclusive_mean_area_eccentricity_excel.py

# Knockout effect dumbbell plots (LL vs LD)
python Static_network_analysis/comparative/knockout/plot_knockout_compare.py

# Build Cytoscape-ready weighted network files
python Static_network_analysis/network/build_weighted_networks.py
```

Outputs:
- `parameter_fold_change_plots/` — per-parameter LL/LD period curves
- `mean_area_eccentricity_plots/` — phase geometry slopes
- `conclusive_mean_area_eccentricity_vs_fold_change.xlsx`
- `network/edges_LL_period.csv`, `nodes_LL_composite.csv`, network PDFs for import into Cytoscape

---

### Natural Light Validation

Run the M1 model under four measured seasonal photoperiods and compare against published expression data.

```bash
python model_with_natural_data/trial1_natural_autumn/m1.py
python model_with_natural_data/trial2_natural_spring/m1.py
python model_with_natural_data/trial3_natural_summer/m1.py
python model_with_natural_data/trial4_natural_winter/m1.py

# Generate comparison plots for each trial
python model_with_natural_data/trial1_natural_autumn/comparison_plots.py
# (repeat for other trials)
```

Each trial reads the corresponding `.xlsx` file (raw and normalised experimental data) in its folder and writes comparison PNGs and `*_extracted_data.csv`.

---

### Hypocotyl Growth Extension

Simulate PIF-dependent hypocotyl growth alongside the core clock oscillation.

```bash
python model/hypocotyl/m1_with_hypo_plot.py
```

Output: growth-rate curves overlaid on clock waveforms.

---

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Waveform PNGs | `model/12l12d/` | All 19 state variables over 480 h |
| Performance metrics | `model1_performance.txt` | MAE and MSE vs experimental data |
| Parameter sweep CSVs | `parameter_analysis_ll/ld/` | Expression profiles per parameter |
| Period sensitivity | `LL_period_outputs/` | Sensitivity slopes, heatmaps |
| Phase portraits | `LL_outputs/` | Convex-hull area and eccentricity |
| Knockout results | `knockout_output_LL/LD/` | Bar plots, waveform overlays, class table |
| Comparative figures | `parameter_fold_change_plots/` | LL vs LD period fold-change |
| Network files | `network/` | Cytoscape edge/node CSVs and PDFs |
| Excel summaries | `Static_network_analysis/` | Integrated .xlsx workbooks |

---

## Model Description

### State Variables (19 total)

| Variable | Biological Identity       |
|----------|--------------------------|
| CLm / CLp | LHY/CCA1 mRNA / protein |
| P97m / P97p | PRR9/7 mRNA / protein |
| P51m / P51p | PRR5/TOC1 mRNA / protein |
| ELm / ELp | ELF4/LUX mRNA / protein |
| PhyA | Phytochrome A (active form) |
| PhyB | Phytochrome B (active form) |
| Cry | Cryptochrome 1 |
| COP1_1/2/3 | COP1 protein complexes |
| PIF | Phytochrome Interacting Factor |
| HYP | Hypocotyl length |
| GZ | Regulatory GZ protein |

### Light Inputs

Two continuous light signals are computed at each time step:
- **Red light (Ired)** — drives PhyA and PhyB activation
- **Blue light (Iblue)** — drives Cry1 activation
- Normalised by factors `eta1` (red) and `eta2` (blue)

### Parameter Sensitivity Classification

After knockout simulation, each parameter is assigned to a class:

| Class | Criterion |
|-------|-----------|
| I (Essential) | Complete arrhythmia on knockout |
| II (Moderate) | Period shift 0.05–5 h |
| III (Minor) | Period shift < 0.05 h |

---

## Data Formats

**Input (experimental, `.xlsx`)**

| Column | Content |
|--------|---------|
| Time | Hours after dawn |
| CCA1, LHY, PRR9, PRR7, PRR5, TOC1, ELF4, LUX | Normalised mRNA expression |

**Expression profile CSVs**

```
Time,LHYm,P97m,P51m,ELm
0.0, 0.83, 0.12, ...
1.0, 0.79, 0.14, ...
```

**Knockout summary CSV (`knockout_analysis_ll.csv`)**

```
Parameter,Value,Label,Period_LHYm,Period_P97m,Period_P51m,Period_ELm
```

---

## Citation

If you use this code or model in your work, please cite the associated manuscript (details to be added upon publication).

---
