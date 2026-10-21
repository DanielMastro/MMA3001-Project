import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

##############################################################################
# USER SETTINGS
##############################################################################

csv_file = r'C:\Users\danma\Documents\school\Monash\Third_Year\Sem_2\NMML\Project\MMA3001-Project\Input_Data\PMSM_2D_FEA_Data.csv'

# Remove initial transient period
init_period_to_remove = 0.5
num_periods = 1.5

Factor2Dto3D = 0.9
# ASSUMPTION:
# TimingAdjustment = 0 deg corresponds to pure Q-axis current
#
# Id = -I*sin(angle)
# Iq =  I*cos(angle)
#
##############################################################################

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def abc_to_alphabeta(a, b, c):

    alpha = (2.0/3.0) * (a - 0.5*b - 0.5*c)

    beta = (2.0/3.0) * ((np.sqrt(3)/2.0) * (b - c))

    return alpha, beta


def park_transform(alpha, beta, theta):

    d = alpha*np.cos(theta) + beta*np.sin(theta)

    q = -alpha*np.sin(theta) + beta*np.cos(theta)

    return d, q


##############################################################################
# LOAD DATA
##############################################################################

print("Loading CSV...")

raw_data = pd.read_csv(csv_file)

##############################################################################
# REMOVE INITIAL PERIOD (SAME STYLE AS YOUR EXISTING CODE)
##############################################################################

n_steps = int(raw_data["NStepsPerElectricalCycle []"].iloc[0])

remove_fraction = init_period_to_remove / num_periods

remove_rows = int(np.floor(n_steps * remove_fraction))

indices = (
    np.arange(len(raw_data))
    .reshape(-1, n_steps + 1)[:, remove_rows:]
    .flatten()
)

data = raw_data.iloc[indices].copy()

print(f"Rows after filtering = {len(data)}")

##############################################################################
# UNIT CONVERSIONS
##############################################################################

# Inductances

data["Laa"] = data["L(Phase_A,Phase_A) [uH]"] * 1e-6 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)
data["Lbb"] = data["L(Phase_B,Phase_B) [uH]"] * 1e-6 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)
data["Lcc"] = data["L(Phase_C,Phase_C) [uH]"] * 1e-6 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)

data["Lab"] = data["L(Phase_A,Phase_B) [nH]"] * 1e-9 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)
data["Lac"] = data["L(Phase_A,Phase_C) [nH]"] * 1e-9 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)

data["Lba"] = data["L(Phase_B,Phase_A) [nH]"] * 1e-9 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)
data["Lbc"] = data["L(Phase_C,Phase_B) [nH]"] * 1e-9 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)

data["Lca"] = data["L(Phase_C,Phase_A) [nH]"] * 1e-9 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)
data["Lcb"] = data["L(Phase_B,Phase_C) [nH]"] * 1e-9 *(33/2)**2 / (np.sqrt(2) * Factor2Dto3D)

# Currents

data["Ia"] = data["InputCurrent(Phase_A) [mA]"] * 1e-3 *(2/33) / (np.sqrt(2) * Factor2Dto3D)
data["Ib"] = data["InputCurrent(Phase_B) [mA]"] * 1e-3 *(2/33) / (np.sqrt(2) * Factor2Dto3D)
data["Ic"] = data["InputCurrent(Phase_C) [mA]"] * 1e-3 *(2/33) / (np.sqrt(2) * Factor2Dto3D)

# Voltages

data["Ea"] = data["InducedVoltage(Phase_A) [uv]"] * 1e-6 *(33/2) / (np.sqrt(2) * Factor2Dto3D)
data["Eb"] = data["InducedVoltage(Phase_B) [uv]"] * 1e-6 *(33/2) / (np.sqrt(2) * Factor2Dto3D)
data["Ec"] = data["InducedVoltage(Phase_C) [uv]"] * 1e-6 *(33/2) / (np.sqrt(2) * Factor2Dto3D)

##############################################################################
# ELECTRICAL ANGLE
#
# One electrical cycle stored as:
# 0, 7.5, 15, ... , 180
#
# We reconstruct electrical angle from step index.
##############################################################################


data["ElectricalAngle_deg"] = (
    data["TimingAdjustment [deg]"]
)

data["ElectricalAngle_rad"] = np.deg2rad(
    data["ElectricalAngle_deg"]
)

##############################################################################
# DQ FLUX LINKAGE
##############################################################################

psi_d = []
psi_q = []

for _, row in data.iterrows():

    alpha, beta = abc_to_alphabeta(
        row["FluxLinkage(Phase_A) [Wb]"],
        row["FluxLinkage(Phase_B) [Wb]"],
        row["FluxLinkage(Phase_C) [Wb]"]
    )

    d, q = park_transform(
        alpha,
        beta,
        row["ElectricalAngle_rad"]
    )

    psi_d.append(d)
    psi_q.append(q)

data["Psi_d"] = psi_d
data["Psi_q"] = psi_q

##############################################################################
# DQ BEMF
##############################################################################

bemf_d = []
bemf_q = []

for _, row in data.iterrows():

    alpha, beta = abc_to_alphabeta(
        row["Ea"],
        row["Eb"],
        row["Ec"]
    )

    d, q = park_transform(
        alpha,
        beta,
        row["ElectricalAngle_rad"]
    )

    bemf_d.append(d)
    bemf_q.append(q)

data["Bemf_d"] = bemf_d
data["Bemf_q"] = bemf_q

##############################################################################
# PARK TRANSFORM THE INDUCTANCE MATRIX
##############################################################################

Ld_matrix = []
Lq_matrix = []

for _, row in data.iterrows():

    theta = row["ElectricalAngle_rad"]

    T = (2/3) * np.array([
        [
            np.cos(theta),
            np.cos(theta - 2*np.pi/3),
            np.cos(theta + 2*np.pi/3)
        ],
        [
            -np.sin(theta),
            -np.sin(theta - 2*np.pi/3),
            -np.sin(theta + 2*np.pi/3)
        ],
        [
            0.5,
            0.5,
            0.5
        ]
    ])

    Labc = np.array([
        [row["Laa"], row["Lab"], row["Lac"]],
        [row["Lba"], row["Lbb"], row["Lbc"]],
        [row["Lca"], row["Lcb"], row["Lcc"]]
    ])

    Ldq0 = T @ Labc @ np.linalg.inv(T) 

    Ld_matrix.append(Ldq0[0,0])
    Lq_matrix.append(Ldq0[1,1])

data["Ld_matrix"] = Ld_matrix 
data["Lq_matrix"] = Lq_matrix 

##############################################################################
# CURRENT MAGNITUDE
##############################################################################

data["I_mag"] = (2/3) * np.sqrt(
    
    (
        data["Ia"]**2 +
        data["Ib"]**2 +
        data["Ic"]**2
    )
)

##############################################################################
# TIMING ANGLE -> ID IQ
#
# USER ASSUMPTION:
# TimingAdjustment=0 = Pure Q axis
##############################################################################

theta_current = np.deg2rad(
    data["TimingAdjustment [deg]"]
)

data["Id_ref"] = (
    -data["I_mag"] *
    np.sin(theta_current)
)

data["Iq_ref"] = (
    data["I_mag"] *
    np.cos(theta_current)
)

##############################################################################
# GROUP TO SINGLE OPERATING POINT
##############################################################################

map_data = data.groupby(
    [
        "J [A_per_m2]",
        "TimingAdjustment [deg]"
    ]
).agg({

    "Id_ref":"mean",
    "Iq_ref":"mean",

    "Psi_d":"mean",
    "Psi_q":"mean",

    "Bemf_d":"mean",
    "Bemf_q":"mean",

    "Ld_matrix":"mean",
    "Lq_matrix":"mean"

}).reset_index()

##############################################################################
# PM FLUX
##############################################################################

pm_points = map_data[
    map_data["J [A_per_m2]"] == 0
]

psi_pm = pm_points["Psi_d"].mean()

print("")
print("Estimated PM Flux")
print(f"{psi_pm:.6f} Wb")
print("")

##############################################################################
# FLUX DERIVED INDUCTANCES
##############################################################################

eps = 1e-9

map_data["Ld_flux"] = np.where(

    np.abs(map_data["Id_ref"]) > eps,

     (
        map_data["Psi_d"] - psi_pm
    )
    /
    map_data["Id_ref"],

    np.nan
)

map_data["Lq_flux"] = np.where(

    np.abs(map_data["Iq_ref"]) > eps,

    map_data["Psi_q"]
    /
    map_data["Iq_ref"],

    np.nan
)

##############################################################################
# TOTAL FLUX MAGNITUDE
##############################################################################

map_data["Psi_mag"] = np.sqrt(
    map_data["Psi_d"]**2 +
    map_data["Psi_q"]**2
) 

##############################################################################
# BEMF MAGNITUDE
##############################################################################

map_data["Bemf_mag"] = np.sqrt(
    map_data["Bemf_d"]**2 +
    map_data["Bemf_q"]**2
)

##############################################################################
# SAVE RESULTS
##############################################################################

map_output_file = (
    Path(__file__).resolve().parent.parent
    / "Output_Data"
    / "dq_motor_map.csv"
)
map_data.to_csv(map_output_file, index=False)

print("Saved:")
print(map_output_file)


##############################################################################
# SUBPLOT: Ld, Lq, Flux Maps
##############################################################################

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# -------------------------
# Ld map
# -------------------------
sc1 = axs[0].scatter(
    map_data["Id_ref"],
    map_data["Iq_ref"],
    c=map_data["Ld_matrix"] * 1e6
)
axs[0].set_title("Ld Map")
axs[0].set_xlabel("Id [A]")
axs[0].set_ylabel("Iq [A]")
axs[0].grid(True)
fig.colorbar(sc1, ax=axs[0], label="Ld [µH]")

# -------------------------
# Lq map
# -------------------------
sc2 = axs[1].scatter(
    map_data["Id_ref"],
    map_data["Iq_ref"],
    c=map_data["Lq_matrix"] * 1e6
)
axs[1].set_title("Lq Map")
axs[1].set_xlabel("Id [A]")
axs[1].set_ylabel("Iq [A]")
axs[1].grid(True)
fig.colorbar(sc2, ax=axs[1], label="Lq [µH]")

# -------------------------
# Flux map
# -------------------------
flux = map_data["Psi_mag"] * (33/2) * (np.sqrt(2) * 0.9)

sc3 = axs[2].scatter(
    map_data["Id_ref"],
    map_data["Iq_ref"],
    c=flux
)
axs[2].set_title("Flux Map")
axs[2].set_xlabel("Id [A]")
axs[2].set_ylabel("Iq [A]")
axs[2].grid(True)
fig.colorbar(sc3, ax=axs[2], label="Flux [Wb]")

plt.tight_layout()
plt.show()

##############################################################################
# SUBPLOT: Ld, Lq, Flux Maps
##############################################################################

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# -------------------------
# Ld map
# -------------------------
sc1 = axs[0].scatter(
    data["Id_ref"],
    data["Iq_ref"],
    c= data["Ld_matrix"] * 1e6
)
axs[0].set_title("Ld Map")
axs[0].set_xlabel("Id [A]")
axs[0].set_ylabel("Iq [A]")
axs[0].grid(True)
fig.colorbar(sc1, ax=axs[0], label="Ld [µH]")

# -------------------------
# Lq map
# -------------------------
sc2 = axs[1].scatter(
    data["Id_ref"],
    data["Iq_ref"],
    c= data["Lq_matrix"] * 1e6
)
axs[1].set_title("Lq Map")
axs[1].set_xlabel("Id [A]")
axs[1].set_ylabel("Iq [A]")
axs[1].grid(True)
fig.colorbar(sc2, ax=axs[1], label="Lq [µH]")

# -------------------------
# Flux map
# -------------------------
flux = map_data["Psi_mag"] * (33/2) * (np.sqrt(2) * 0.9)

sc3 = axs[2].scatter(
    map_data["Id_ref"],
    map_data["Iq_ref"],
    c=flux
)
axs[2].set_title("Flux Map")
axs[2].set_xlabel("Id [A]")
axs[2].set_ylabel("Iq [A]")
axs[2].grid(True)
fig.colorbar(sc3, ax=axs[2], label="Flux [Wb]")

plt.tight_layout()
plt.show()

##############################################################################
# BUILD EXPORT TABLE
##############################################################################

export_df = map_data.copy()

# Optional: add derived flux column (same as your plot)
export_df["Flux_Wb"] = export_df["Psi_mag"] * (33/2)**2 

# Optional: scale inductances to µH for CSV readability
export_df["Ld_uH"] = export_df["Ld_matrix"] * 1e6
export_df["Lq_uH"] = export_df["Lq_matrix"] * 1e6

# Keep only relevant columns (clean output)
export_df = export_df[[
    "Id_ref",
    "Iq_ref",
    "Ld_uH",
    "Lq_uH",
    "Flux_Wb"
]]

##############################################################################
# SAVE TO CSV
##############################################################################

output_file = (
    Path(__file__).resolve().parent.parent
    / "Output_Data"
    / "PMSM_2D_FEA_Map.csv"
)
export_df.to_csv(output_file, index=False)

print(f"Saved: \n PMSM_2D_FEA_Map.csv")