
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from math import log

FIGSIZE = (3, 2)
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 6
LEGEND_SIZE = 6
TITLE_SIZE = 8
LINE_WIDTH = 1.3
SPINE_WIDTH = 0.8
EXPORT_DPI = 600


def create_styled_axes():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(
        top=0.90,
        bottom=0.28,
        left=0.20,
        right=0.97,
        hspace=0.2,
        wspace=0.2,
    )
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE, width=SPINE_WIDTH)
    ax.margins(x=0.04)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    return fig, ax

def dCdt(t, C):


    v1 = 4.8318   #CL synthesis
    q1a = 1.4266  #CL light-induced synthesis through PhyA
    q3a = 8.9432  #CL light-induced synthesis through PhyB
    eta1 = 0.03  #Normalisation of red light intensity
    q4a = 5.9277  #CL light-induced synthesis through Cry
    eta2 = 0.0215  #Normalisation of blue light intensity
    K1 = 0.1943  #Inhibition: CL by P97
    K2 = 1.6138  #Inhibition: CL by P51
    k1L = 0.2866 #CL mRNA degradation (light)
    k1D = 0.213  #CL mRNA degradation (dark)
    p1 = 0.8672 #CL translation
    p1L = 0.2378 #CL light-induced translation
    d1 = 0.7843  #CL degradation
    q1b = 3.575  #P97 light-induced synthesis through PhyA
    q3b = 5.5899 #P97 light-induced synthesis through PhyB
    q4b = 8.954  #P97 light-induced synthesis through Cry
    v2 = 1.6822    #P97 synthesis
    K3 = 2.2275 #Inhibition: P97 by CL
    K4 = 0.40  #Inhibition: P97 by P51
    K5 = 0.37 #Inhibition: P97 by EL
    k2 = 0.35  #P97 mRNA degradation
    p2 = 0.7858  #P97 translation
    d2D = 0.3712  #P97 degradation (dark)
    d2L = 0.2917  #P97 degradation (light)
    v3 = 1.113  #P51 synthesis
    K6 = 0.4944  #Inhibition: P51 by CL
    K7 = 2.4087  #Inhibition: P51 by itself
    k3 = 0.5819  #P51 mRNA degradation
    p3 = 0.6142 #P51 translation
    d3D = 0.5026  #P51 degradation (dark)
    d3L = 0.5431  #P51 degradation (light)
    v4 = 2.5012   #EL synthesis
    K8 = 0.3262  #Inhibition: EL by CL
    K9 = 1.7974  #Inhibition: EL by P51
    K10 = 1.1889  #Inhibition: EL by EL
    k4 = 0.925  #EL mRNA degradation
    p4 = 1.126  #EL translation
    de1 = 0.0022 #EL degradation
    de2 = 0.4741  #EL degradation (COP1)
    de3 = 0.3765  #EL degradation (COP1: PhyA)
    de4 = 0.398  #EL degradation (COP1: PhyB)
    de5 = 0.0003  #EL degradation (COP1: Cry)
    Ap3 = 0.3868 #PhyA translation
    Am7 = 0.5503 #PhyA degradation
    Ak7 = 1.125  #Michaelis constant of PhyA degradation
    q2 = 0.5767  #Rate constant of light-independent
    kmpac = 137 #Binding rate of COP1: PhyA
    kd = 7  #Dissociate rate
    v5 = 0.1129  #PIF synthesis
    K11 = 0.3322  #Inhibition: PIF by EL
    k5 = 0.1591  #PIF mRNA degradation
    p5 = 0.5293  #PIF translation
    d5D = 0.4404  #PIF protein degradation (Dark)
    d5L = 5.0712  #PIF protein degradation (Light)
    g1 = 0.001  #Baseline hypocotyl growth
    g2 = 0.18  #PIF-induced hypocotyl growth
    K12 = 0.86  #Activation: growth by PIF
    Bp4 = 0.4147  #PhyB translation
    Bm8 = 0.7728 #PhyB degradation
    Bk8 = 0.1732  #Michaelis constant of PhyB degradation
    kmpbc = 7162  #Binding rate of COP1:PhyB
    Cp5 = 0.4567 #Cry translation
    Cm9 = 0.867  #Cry degraddation
    Ck9 = 0.3237  #Michaelis constant of Cry degradation
    kmcc = 13406  #Binding rate of COP1:Cry
    K14 = 1.5    #Inhibition: PIF by Cry
    Gp6 = 0.000100 #GZ translation
    dg1 = 0.010000 #GZ degradation
    dg2 = 1.280202 #GZ degradation (COP1)
    dg3 = 0.010000 #GZ degradation (COP1:PhyA)
    dg4 = 1.750462 #GZ degradation (COP1:PhyB)
    dg5 = 1.067661 #GZ degradation (COP1:Cry)
    dp6 = 0.010000 #P51 degradation (GZ)
    Gkp = 1.185527 #Michaelis constant of P51 degradation


    if Ired != 0 or Iblue != 0:
        ThetaPhyA = 1
    else:
        ThetaPhyA = 0

    if Ired != 0:
        ThetaPhyB = 1
    else:
        ThetaPhyB = 0

    if Iblue != 0:
        ThetaCry1 = 1
    else:
        ThetaCry1 = 0


    dC = np.zeros((19))

    # LHY mRNA
    dC[0] = (v1 + (q1a * (C[8]) * ThetaPhyA + q3a * (C[12]) * log(eta1 * Ired + 1) * ThetaPhyB + q4a * (C[13]) * log(eta2 * Iblue + 1) * ThetaCry1)) / (1 + (C[3] / K1) ** 2 + (C[5] / K2) ** 2) - (k1L * ThetaPhyA + k1D * (1 - ThetaPhyA)) * C[0]

    # LHY protein
    dC[1] = (p1 + p1L * ThetaPhyA) * C[0] - d1 * C[1]

    # P97 mRNA
    dC[2] = ((q1b * (C[8]) * ThetaPhyA + q3b * (C[12]) * log(eta1 * Ired + 1) * ThetaPhyB + q4b * (C[13]) * log(eta2 * Iblue + 1) * ThetaCry1) + v2) * (1 / (1 + (C[1] / K3) ** 2 + (C[5] / K4) ** 2 + (C[7] / K5) ** 2)) - k2 * C[2]

    # P97 protein
    dC[3] = p2 * C[2] - d2D * (1 - ThetaPhyA) * C[3] - d2L * ThetaPhyA * C[3]

    # P51 mRNA
    dC[4] = v3 / (1 + (C[1] / K6) ** 2 + (C[5] / K7) ** 2) - k3 * C[4]

    # P51 protein
    dC[5] = (p3 * C[4]) - (d3D * (1 - ThetaPhyA) * C[5]) - (d3L * ThetaPhyA * C[5]) - ((dp6 * C[18]* C[5])/(Gkp + C[5]))

    # EL mRNA
    dC[6] = (v4 * ThetaPhyA / (1 + (C[1] / K8) ** 2 + (C[5] / K9) ** 2 + (C[7] / K10) ** 2) - k4 * C[6])

    # EL protein
    dC[7] = (p4 * C[6] - (de1 + (de2 * C[14] + de3 * C[15] + de4 * C[16] + de5 * C[17]) / (C[14] + C[15] + C[16] + C[17])) * C[7])

    # PHY A
    dC[8] = (1 - ThetaPhyA) * Ap3 - (Am7 * C[8] / (Ak7 + C[8])) - q2 * ThetaPhyA * C[8] - kmpac * ThetaPhyA * C[8] * C[14] + kd * C[15]

    # PIF mRNA
    dC[9] = v5 / (1 + (C[7] / K11) ** 2 + (C[13]/K14) ** 2) - k5 * C[9]

    # PIF protein
    dC[10] = p5 * C[9] - d5D * (1 - ThetaPhyA) * C[10] - d5L * ThetaPhyA * C[10]

    # HYP protein
    dC[11] = g1 + (g2 * C[10] ** 2) / (K12 ** 2 + C[10] ** 2)

    # PHY B
    dC[12] = Bp4 - ((Bm8 * C[12]) / (Bk8 + C[12])) - kmpbc * ThetaPhyB * C[12] * C[14] + kd * C[16]

    # CRY1
    dC[13] = Cp5 - ((Cm9 * C[13]) / (Ck9 + C[13])) - kmcc * ThetaCry1 * C[13] * C[14] + kd * C[17]

    # COP1
    dC[14] = -kmpac * ThetaPhyA * C[8] * C[14] + kd * C[15] - kmpbc * ThetaPhyB * C[12] * C[14] + kd * C[16] - kmcc * ThetaCry1 * C[13] * C[14] + kd * C[17] + (
            Am7 * C[15] / (Ak7 + C[15])) + q2 * ThetaPhyA * C[15] + ((Bm8 * C[16]) / (Bk8 + C[16])) + ((Cm9 * C[17]) / (Ck9 + C[17]))

    # COP1:PHYA
    dC[15] = kmpac * ThetaPhyA * C[8] * C[14] - kd * C[15] - (Am7 * C[15] / (Ak7 + C[15])) - q2 * ThetaPhyA * C[15]

    # COP1:PHYB
    dC[16] = kmpbc * ThetaPhyB * C[12] * C[14] - kd * C[16] - ((Bm8 * C[16]) / (Bk8 + C[16]))

    # COP1:CRY1
    dC[17] = kmcc * ThetaCry1 * C[13] * C[14] - kd * C[17] - ((Cm9 * C[17]) / (Ck9 + C[17]))

    #GZ protein
    dC[18] = Gp6 -  (dg1 + (dg2 * C[14] + dg3 * C[15] + dg4 * C[16] + dg5 * C[17]) / (C[14] + C[15] + C[16] + C[17])) * C[18]


    return dC


Ired = 0
Iblue = 0
eta1 = 0
eta2 = 0


pPlot = []

Nday = 10  # Experimental data collected at Day 10

hyplength = np.zeros((3, 3))

IntensityBB = []
IntensityRR = []

for LH in range(18, 23, 2):  # Photoperiod (hours in darkness)
    for wave in range(1, 4):  # Light colors
        # Set the Intensity to 26.62 for ON and 0 for OFF

        if wave == 1:
            # Blue light
            IntensityBB = np.concatenate((np.ones(24-LH), np.zeros(LH))) * 26.62
            IntensityRR = np.concatenate((np.ones(24-LH), np.zeros(LH))) * 0

        elif wave == 2:
            # Red light
            IntensityBB = np.concatenate((np.ones(24-LH), np.zeros(LH))) * 0
            IntensityRR = np.concatenate((np.ones(24-LH), np.zeros(LH))) * 26.62

        else:
            # Mixed
            IntensityBB = np.concatenate((np.ones(24-LH), np.zeros(LH))) * 26.62
            IntensityRR = np.concatenate((np.ones(24-LH), np.zeros(LH))) * 26.62


        temp1 = np.tile(IntensityBB, Nday)
        IntensityBB = np.transpose(temp1)
        temp2 = np.tile(IntensityRR, Nday)
        IntensityRR = np.transpose(temp2)


        ProteinLevel = []
        C = 1*np.ones((1,19))  # Initial conditions for the 19 variables
        C[0,11] = 0
        C[0,15] = 0
        C[0,16] = 0
        C[0,17] = 0

#providing different light conditions
        for i in range(len(IntensityRR)):
            tspan = [i, i + 1]
            Ired = IntensityRR[i]
            Iblue = IntensityBB[i]



           #solving ODEs
            sol = odeint(dCdt, C[-1], tspan, tfirst=True)

            C=np.vstack((C,sol[-1]))



            #stacking final values of protein expression to protein level
            if len(ProteinLevel) == 0:
                ProteinLevel = C[-1]
            else:
                ProteinLevel = np.vstack((ProteinLevel, C[-1]))
        
        a = (LH // 2 - 9)
        b = (wave - 1)
        hyplength[a, b] = ProteinLevel[239, 11]  # Calculating hypocotyl length



# Simulated Hypocotyl Length Plotting
h = hyplength.tolist()
S_hyp = np.vstack((h[0], h[1], h[2]))
print(S_hyp)


x_labels = ['6L18D', '4L20D', '2L22D']
colors = ['blue', 'red', 'purple']
legend_labels = ['Blue', 'Red', 'Blue+Red']

fig, ax = create_styled_axes()
bar_width = 0.2
for i in range(3):
    ax.bar(
        np.arange(3) + i * bar_width,
        S_hyp[:, i],
        width=bar_width,
        align='center',
        label=legend_labels[i],
        color=colors[i],
        linewidth=LINE_WIDTH,
    )

ax.set_ylim((0, 20))
ax.set_ylabel('Hypocotyl Length (mm)', fontsize=AXIS_LABEL_SIZE)
ax.set_xlabel('Light Duration (h)', fontsize=AXIS_LABEL_SIZE)
ax.set_xticks(np.arange(3), x_labels)
ax.legend(fontsize=LEGEND_SIZE)
ax.set_title('Simulated Hypocotyl Length', fontsize=TITLE_SIZE, pad=2)
fig.savefig('Simulated Hypocotyl Length.jpg', dpi=EXPORT_DPI, bbox_inches='tight', pad_inches=0.02)

# Measured Hypocotyl Length Plotting
M_hyp = np.array([[5.13, 7.01, 5.59], [6.14, 9.42, 6.95], [8.77, 10.6, 9.2]])
sd_hyp = np.array([[0.87, 2.05, 1], [1.14, 1.72, 1.19], [2.60, 2.18, 1.57]])

fig, ax = create_styled_axes()
bar_width = 0.2
for i in range(3):
    ax.bar(
        np.arange(3) + i * bar_width,
        M_hyp[:, i],
        width=bar_width,
        align='center',
        label=legend_labels[i],
        color=colors[i],
        linewidth=LINE_WIDTH,
    )
    ax.errorbar(
        np.arange(3) + i * bar_width,
        M_hyp[:, i],
        sd_hyp[:, i],
        fmt='.',
        color='black',
        linewidth=LINE_WIDTH,
    )

ax.set_ylim((0, 20))
ax.set_ylabel('Hypocotyl Length (mm)', fontsize=AXIS_LABEL_SIZE)
ax.set_xlabel('Light Duration (h)', fontsize=AXIS_LABEL_SIZE)
ax.set_xticks(np.arange(3), x_labels)
ax.legend(fontsize=LEGEND_SIZE)
ax.set_title('Measured Hypocotyl Length', fontsize=TITLE_SIZE, pad=2)
fig.savefig('Measured Hypocotyl Length.jpg', dpi=EXPORT_DPI, bbox_inches='tight', pad_inches=0.02)

# Row-wise min-max scaling using S_hyp directly
LH1 = [(x - S_hyp[0].min()) * ((M_hyp[0].max() - M_hyp[0].min()) / (S_hyp[0].max() - S_hyp[0].min())) + M_hyp[0].min() for x in S_hyp[0]]
LH2 = [(x - S_hyp[1].min()) * ((M_hyp[1].max() - M_hyp[1].min()) / (S_hyp[1].max() - S_hyp[1].min())) + M_hyp[1].min() for x in S_hyp[1]]
LH3 = [(x - S_hyp[2].min()) * ((M_hyp[2].max() - M_hyp[2].min()) / (S_hyp[2].max() - S_hyp[2].min())) + M_hyp[2].min() for x in S_hyp[2]]

N_hyp = np.array([LH1, LH2, LH3])

fig, ax = create_styled_axes()
bar_width = 0.2
for i in range(3):
    ax.bar(
        np.arange(3) + i * bar_width,
        N_hyp[:, i],
        width=bar_width,
        align='center',
        label=legend_labels[i],
        color=colors[i],
        linewidth=LINE_WIDTH,
    )

ax.set_ylim((0, 20))
ax.set_ylabel('Hypocotyl Length (mm)', fontsize=AXIS_LABEL_SIZE)
ax.set_xlabel('Light Duration (h)', fontsize=AXIS_LABEL_SIZE)
ax.set_xticks(np.arange(3), x_labels)
ax.legend(fontsize=LEGEND_SIZE)
ax.set_title('Simulated Hypocotyl Length (Normalized)', fontsize=TITLE_SIZE, pad=2)
fig.savefig('Simulated Hypocotyl Length (Normalized).jpg', dpi=EXPORT_DPI, bbox_inches='tight', pad_inches=0.02)

plt.show()
