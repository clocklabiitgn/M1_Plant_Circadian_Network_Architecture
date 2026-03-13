# M1 Circadian Network Architecture

Code and processed data for the manuscript:

**Dissecting the Network Architecture of a Circadian Clock Model: Identifying Key Regulatory Mechanisms and Essential Interactions**

This repository contains the M1 Arabidopsis circadian clock model and the analysis workflows used to quantify network architecture under constant light (LL) and light-dark (LD) conditions. The workflows cover parameter-range simulations, knockout analysis, period sensitivity analysis, phase-portrait analysis, and LL-vs-LD comparative network construction.

## Repository overview

```text
M1_Circadian_Network_Architecture/
├── model/
│   └── m1_model.py
├── analysis/
│   ├── ll/
│   │   ├── parameter_range_auto_ll.py
│   │   ├── knockout_analysis/
│   │   ├── period_senstivity_analysis/
│   │   └── phase_potrait_analysis/
│   ├── ld/
│   │   ├── parameter_range_auto_ld.py
│   │   ├── knockout_analysis/
│   │   ├── period_senstivity_analysis/
│   │   └── phase_potrait_analysis/
│   └── comparative/
│       ├── compare_knockout_ll_ld.py
│       ├── compare_period_senstivity_ll_ld.py
│       ├── compare_phase_potrait_ll_ld.py
│       └── build_weighted_networks.py
├── environment.yml
└── .gitignore
```

## What this repository contains

### 1. Core model
`model/m1_model.py` contains the M1 ODE-based circadian clock model implementation.

### 2. LL and LD analysis workflows
The `analysis/ll/` and `analysis/ld/` folders contain condition-specific scripts for:
- parameter-range simulations
- knockout analysis
- period sensitivity analysis
- phase-portrait generation and geometric analysis

### 3. Comparative analysis
The `analysis/comparative/` folder contains scripts that compare LL and LD outputs and build weighted directed networks from the integrated results.

## Suggested execution order

### Constant light (LL)
1. `analysis/ll/parameter_range_auto_ll.py`
2. `analysis/ll/knockout_analysis/knockout_ll.py`
3. `analysis/ll/knockout_analysis/plot_knockout.py`
4. `analysis/ll/period_senstivity_analysis/1_period_analysis.py`
5. `analysis/ll/period_senstivity_analysis/2_calculate_period_change.py`
6. `analysis/ll/period_senstivity_analysis/3_plot_period_change.py`
7. `analysis/ll/phase_potrait_analysis/phase_and_expression_plots_per_value.py`
8. `analysis/ll/phase_potrait_analysis/phase_plot_frame_analyzer.py`

### Light-dark (LD)
1. `analysis/ld/parameter_range_auto_ld.py`
2. `analysis/ld/knockout_analysis/knockout_ld.py`
3. `analysis/ld/knockout_analysis/plot_knockout.py`
4. `analysis/ld/period_senstivity_analysis/1_period_analysis.py`
5. `analysis/ld/period_senstivity_analysis/2_calculate_period_change.py`
6. `analysis/ld/period_senstivity_analysis/3_plot_period_change.py`
7. `analysis/ld/phase_potrait_analysis/phase_and_expression_plots_per_value.py`
8. `analysis/ld/phase_potrait_analysis/phase_plot_frame_analyzer.py`

### Comparative integration
1. `analysis/comparative/compare_knockout_ll_ld.py`
2. `analysis/comparative/compare_period_senstivity_ll_ld.py`
3. `analysis/comparative/compare_phase_potrait_ll_ld.py`
4. `analysis/comparative/build_weighted_networks.py`

## Environment setup

```bash
conda env create -f environment.yml
conda activate m1-circadian-network
```


## Main Python dependencies
- numpy
- scipy
- pandas
- matplotlib
- seaborn
- networkx
- openpyxl
- pillow
- opencv-python
- tqdm

## Contact
**Shashank Kumar Singh**  
Department of Biological Sciences and Engineering  
Indian Institute of Technology Gandhinagar
