import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# USER SETTINGS
# ============================================================================

csv_file = r'C:\Users\danma\Documents\school\Monash\Third_Year\Sem_2\NMML\Project\MMA3001-Project\Input_Data\PMSM_2D_FEA_Data.csv'

turns_ratio   = 33 / 2     # N_real / N_model  (linear-in-turns for flux/voltage)
stack_factor  = 0.9        # 2D -> 3D correction, applied to field quantities only 
pole_pairs    = 5          # <-- SET THIS. Needed for torque. Guessing wrecks torque.

# How to get the rotor electrical angle for the Park transform.
#   'from_flux' : read it from the no-load (J=0) flux direction. At zero current
#                 the flux linkage points along the d-axis, so its angle IS the
#                 rotor position at each step. Works for a rotating rotor (incl.
#                 the 0.000001 RPM quasi-static sweep) with no need to know the
#                 span or reference offset. RECOMMENDED.
#   'fixed'     : rotor held at one position (constant angle + auto-align).
#   'sweep'     : reconstruct angle geometrically from step index (needs the two
#                 settings below to be correct).
rotor_mode      = 'from_flux'
rotor_ref_deg   = 0.0      # only used by 'fixed'/'sweep': d-axis reference
auto_align      = True     # 'fixed' only: snap so no-load psi_q = 0

# Only used when rotor_mode == 'sweep'
init_period_to_remove = 0.5
num_periods           = 1.5
elec_deg_per_cycle    = 360.0   # electrical degrees spanned by one stored cycle

# ============================================================================
# PARK / CLARKE (amplitude-invariant, 2/3 convention)
# ============================================================================
def park_vec(a, b, c, theta):
    """abc -> dq0 for a vector quantity (flux, current, voltage)."""
    th = theta
    d = (2/3) * (a*np.cos(th) + b*np.cos(th - 2*np.pi/3) + c*np.cos(th + 2*np.pi/3))
    q = -(2/3) * (a*np.sin(th) + b*np.sin(th - 2*np.pi/3) + c*np.sin(th + 2*np.pi/3))
    z = (1/3) * (a + b + c)
    return d, q, z


def park_matrix(theta):
    return (2/3) * np.array([
        [np.cos(theta), np.cos(theta - 2*np.pi/3), np.cos(theta + 2*np.pi/3)],
        [-np.sin(theta), -np.sin(theta - 2*np.pi/3), -np.sin(theta + 2*np.pi/3)],
        [0.5, 0.5, 0.5],
    ])


def dq_inductances(row, theta):
    """Transform the abc inductance matrix into the dq frame at rotor angle theta.
    Returns incremental Ld, Lq and the dq cross-coupling Ldq (should be ~0 when
    the frame is aligned; large values flag saturation cross-coupling)."""
    T = park_matrix(theta)
    Labc = np.array([
        [row["Laa"], row["Lab"], row["Lac"]],
        [row["Lba"], row["Lbb"], row["Lbc"]],
        [row["Lca"], row["Lcb"], row["Lcc"]],
    ])
    Ldq0 = T @ Labc @ np.linalg.inv(T)
    return Ldq0[0, 0], Ldq0[1, 1], Ldq0[0, 1]


# ============================================================================
# LOAD
# ============================================================================
print("Loading CSV...")
raw = pd.read_csv(csv_file)

# ---- optional transient removal (rotating-rotor case only) ----
if rotor_mode == 'sweep':
    n_steps = int(raw["NStepsPerElectricalCycle []"].iloc[0])
    remove_fraction = init_period_to_remove / num_periods
    remove_rows = int(np.floor(n_steps * remove_fraction))
    idx = (np.arange(len(raw))
           .reshape(-1, n_steps + 1)[:, remove_rows:]
           .flatten())
    data = raw.iloc[idx].copy()
else:
    data = raw.copy()

print(f"Rows = {len(data)}")

# ============================================================================
# RAW COLUMN -> WORKING COLUMNS (no scaling yet, just unit prefixes)
# ============================================================================
# Inductances to Henries
data["Laa"] = data["L(Phase_A,Phase_A) [uH]"] * 1e-6
data["Lbb"] = data["L(Phase_B,Phase_B) [uH]"] * 1e-6
data["Lcc"] = data["L(Phase_C,Phase_C) [uH]"] * 1e-6
data["Lab"] = data["L(Phase_A,Phase_B) [nH]"] * 1e-9
data["Lac"] = data["L(Phase_A,Phase_C) [nH]"] * 1e-9
data["Lba"] = data["L(Phase_B,Phase_A) [nH]"] * 1e-9
data["Lbc"] = data["L(Phase_C,Phase_B) [nH]"] * 1e-9   # reciprocal, name kept from original
data["Lca"] = data["L(Phase_C,Phase_A) [nH]"] * 1e-9
data["Lcb"] = data["L(Phase_B,Phase_C) [nH]"] * 1e-9

# Currents to Amps (raw, model turns)
data["ia"] = data["InputCurrent(Phase_A) [mA]"] * 1e-3
data["ib"] = data["InputCurrent(Phase_B) [mA]"] * 1e-3
data["ic"] = data["InputCurrent(Phase_C) [mA]"] * 1e-3

# Flux linkage already in Wb (raw, model turns)
data["fa"] = data["FluxLinkage(Phase_A) [Wb]"]
data["fb"] = data["FluxLinkage(Phase_B) [Wb]"]
data["fc"] = data["FluxLinkage(Phase_C) [Wb]"]

# ============================================================================
# ROTOR ELECTRICAL ANGLE
# ============================================================================
if rotor_mode == 'from_flux':
    # Read the rotor angle per step from the no-load (J=0) flux direction.
    n_steps = int(raw["NStepsPerElectricalCycle []"].iloc[0])
    block = n_steps + 1
    nl = raw[raw["J [A_per_m2]"] == 0]
    if len(nl) == 0 or len(nl) % block != 0:
        raise ValueError(
            f"from_flux needs whole no-load blocks. Found {len(nl)} J=0 rows, "
            f"block size = {block}. Check NStepsPerElectricalCycle / that J=0 was swept.")
    a0 = nl["FluxLinkage(Phase_A) [Wb]"].to_numpy()
    b0 = nl["FluxLinkage(Phase_B) [Wb]"].to_numpy()
    c0 = nl["FluxLinkage(Phase_C) [Wb]"].to_numpy()
    d0, q0, _ = park_vec(a0, b0, c0, 0.0)                 # -> stationary alpha, beta
    ang = np.unwrap(np.arctan2(q0, d0).reshape(-1, block), axis=1)
    theta_step = np.angle(np.exp(1j * ang).mean(axis=0))  # circular mean per step
    if len(data) % block != 0:
        raise ValueError("Row count is not a whole number of blocks; check the export.")
    data["theta"] = np.tile(theta_step, len(data) // block)

elif rotor_mode == 'sweep':
    n_steps = int(raw["NStepsPerElectricalCycle []"].iloc[0])
    remove_rows = int(np.floor(n_steps * init_period_to_remove / num_periods))
    kept = n_steps + 1 - remove_rows
    step_in_cycle = np.tile(np.arange(remove_rows, n_steps + 1),
                            len(data) // kept + 1)[:len(data)]
    data["theta"] = np.deg2rad(rotor_ref_deg) + step_in_cycle * np.deg2rad(elec_deg_per_cycle / n_steps)

else:  # 'fixed'
    data["theta"] = np.full(len(data), np.deg2rad(rotor_ref_deg))
    if auto_align:
        nl = data[data["J [A_per_m2]"] == 0]
        d0, q0, _ = park_vec(nl["fa"].to_numpy(), nl["fb"].to_numpy(),
                             nl["fc"].to_numpy(), nl["theta"].to_numpy())
        delta = np.arctan2(np.mean(q0), np.mean(d0))
        data["theta"] = data["theta"] + delta
        print(f"Auto-align: rotor d-axis shifted by {np.rad2deg(delta):+.3f} deg")

# ============================================================================
# PER-ROW dq QUANTITIES
# ============================================================================
id_r, iq_r, pd_r, pq_r, Ld_r, Lq_r = ([] for _ in range(6))
for _, row in data.iterrows():
    th = row["theta"]
    d_i, q_i, _ = park_vec(row["ia"], row["ib"], row["ic"], th)
    d_f, q_f, _ = park_vec(row["fa"], row["fb"], row["fc"], th)
    Ld, Lq, _ = dq_inductances(row, th)
    id_r.append(d_i); iq_r.append(q_i)
    pd_r.append(d_f); pq_r.append(q_f)
    Ld_r.append(Ld);  Lq_r.append(Lq)

data["id_raw"]  = id_r
data["iq_raw"]  = iq_r
data["psi_d_raw"] = pd_r
data["psi_q_raw"] = pq_r
data["Ld_raw"]  = Ld_r
data["Lq_raw"]  = Lq_r

# ============================================================================
# COLLAPSE EACH OPERATING POINT (average over the stored rows)
# ============================================================================
m = data.groupby(["J [A_per_m2]", "TimingAdjustment [deg]"]).agg(
    id_raw=("id_raw", "mean"),
    iq_raw=("iq_raw", "mean"),
    psi_d_raw=("psi_d_raw", "mean"),
    psi_q_raw=("psi_q_raw", "mean"),
    Ld_raw=("Ld_raw", "mean"),
    Lq_raw=("Lq_raw", "mean"),
).reset_index()

# ============================================================================
# APPLY CONSISTENT TURNS / STACK SCALING
# ============================================================================
N, s = turns_ratio, stack_factor
m["Id"]    = m["id_raw"]    * (1 / N)
m["Iq"]    = m["iq_raw"]    * (1 / N)
m["Psi_d"] = m["psi_d_raw"] * (N * s)
m["Psi_q"] = m["psi_q_raw"] * (N * s)
m["Ld"]    = m["Ld_raw"]    * (N**2 * s)
m["Lq"]    = m["Lq_raw"]    * (N**2 * s)
m["Psi_mag"] = np.hypot(m["Psi_d"], m["Psi_q"])   # total flux linkage magnitude |psi|

# ============================================================================
# PM FLUX (no-load psi_d)
# ============================================================================
psi_pm = m.loc[m["J [A_per_m2]"] == 0, "Psi_d"].mean()
psi_q_nl = m.loc[m["J [A_per_m2]"] == 0, "Psi_q"].abs().mean()
print(f"\npsi_pm       = {psi_pm:.6f} Wb   (peak phase PM flux linkage)")
print(f"no-load |psi_q| = {psi_q_nl:.2e} Wb   (should be ~0 if aligned)")

# ============================================================================
# TORQUE  (straight from the flux map; turns ratio cancels, as it physically must)
# ============================================================================
m["Torque_Nm"] = 1.5 * pole_pairs * (m["Psi_d"] * m["Iq"] - m["Psi_q"] * m["Id"])

# ============================================================================
# QUICK DIAGNOSTICS
# ============================================================================
print("\nSaliency check (incremental):")
print(f"  Ld mean {m.Ld.mean()*1e6:8.2f} uH   Lq mean {m.Lq.mean()*1e6:8.2f} uH")
print(f"  Lq/Ld ratio (mean) = {m.Lq.mean()/m.Ld.mean():.3f}")

# Does Ld-Lq still depend on current angle at fixed |I|?  (it shouldn't now)
m["Imag"] = np.hypot(m.Id, m.Iq)
m["beta"] = np.degrees(np.arctan2(-m.Id, m.Iq))
print("\n(Ld-Lq) in uH vs current angle beta, one |I| band:")
band = m[(m.Imag > 40) & (m.Imag < 55)].sort_values("beta")
print("  " + "  ".join(f"{b:3.0f}:{(l-q)*1e6:+5.1f}"
                        for b, l, q in zip(band.beta, band.Ld, band.Lq)))

# ============================================================================
# EXPORT
# ============================================================================
out = m[[
    "Id", "Iq",
    "Psi_d", "Psi_q", "Psi_mag",
    "Ld", "Lq",
    "Torque_Nm",
]].copy()
out.insert(0, "Psi_pm", psi_pm)
out["Ld_uH"] = out["Ld"] * 1e6
out["Lq_uH"] = out["Lq"] * 1e6
output_file = (
    Path(__file__).resolve().parent.parent
    / "Output_Data"
    / "PMSM_2D_FEA_Map.csv"
)
out.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")

# ============================================================================
# PLOTS
# ============================================================================
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
panels = [("Ld_uH", "Ld [uH]"), ("Lq_uH", "Lq [uH]"),
          ("Psi_mag", "|psi| [Wb]"), ("Torque_Nm", "Torque [Nm]")]
for a, (col, lab) in zip(ax.flat, panels):
    sc = a.scatter(out["Id"], out["Iq"], c=out[col])
    a.set_title(lab); a.set_xlabel("Id [A]"); a.set_ylabel("Iq [A]"); a.grid(True)
    fig.colorbar(sc, ax=a, label=lab)
plt.tight_layout()
figure_file = (
    Path(__file__).resolve().parent.parent
    / "Output_Data"
    / "Motor_Characteristic_Plots.png"
)
fig.savefig(figure_file, dpi=300, bbox_inches="tight")
print(f"Saved: {figure_file}")
plt.show()
