import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# USER SETTINGS
# ============================================================================

csv_file = r'C:\Users\danma\Documents\school\Monash\Third_Year\Sem_2\NMML\Project\MMA3001-Project\Input_Data\PMSM_2D_FEA_Data.csv'

turns_ratio   = 33 / 2     # N_real / N_model
stack_factor  = 0.9        # 2D -> 3D correction, field quantities only
pole_pairs    = 5

# Rotor electrical angle source. 'from_flux' reads it per step from the
# no-load (J=0) flux direction and needs no assumptions about span or offset.
rotor_mode      = 'from_flux'      # 'from_flux' | 'fixed' | 'sweep'
rotor_ref_deg   = 0.0              # 'fixed'/'sweep' only
auto_align      = True             # 'fixed' only

# 'sweep' mode only (legacy fallback)
init_period_to_remove = 0.5
num_periods           = 1.5
elec_deg_per_cycle    = 360.0

# ============================================================================
# PARK / CLARKE (amplitude-invariant, 2/3 convention)
# ============================================================================
def park_vec(a, b, c, theta):
    d = (2/3) * (a*np.cos(theta) + b*np.cos(theta - 2*np.pi/3) + c*np.cos(theta + 2*np.pi/3))
    q = -(2/3) * (a*np.sin(theta) + b*np.sin(theta - 2*np.pi/3) + c*np.sin(theta + 2*np.pi/3))
    z = (1/3) * (a + b + c)
    return d, q, z


def park_matrix(theta):
    return (2/3) * np.array([
        [np.cos(theta), np.cos(theta - 2*np.pi/3), np.cos(theta + 2*np.pi/3)],
        [-np.sin(theta), -np.sin(theta - 2*np.pi/3), -np.sin(theta + 2*np.pi/3)],
        [0.5, 0.5, 0.5],
    ])


def dq_inductances_matrix(row, theta):
    """dq transform of the FEA's abc inductance matrix. DIAGNOSTIC ONLY —
    see module docstring."""
    T = park_matrix(theta)
    L = np.array([
        [row["Laa"], row["Lab"], row["Lac"]],
        [row["Lba"], row["Lbb"], row["Lbc"]],
        [row["Lca"], row["Lcb"], row["Lcc"]],
    ])
    D = T @ L @ np.linalg.inv(T)
    return D[0, 0], D[1, 1]


# ============================================================================
# LOAD
# ============================================================================
print("Loading CSV...")
raw = pd.read_csv(csv_file)

if rotor_mode == 'sweep':
    n_steps = int(raw["NStepsPerElectricalCycle []"].iloc[0])
    remove_rows = int(np.floor(n_steps * init_period_to_remove / num_periods))
    idx = (np.arange(len(raw))
           .reshape(-1, n_steps + 1)[:, remove_rows:]
           .flatten())
    data = raw.iloc[idx].copy()
else:
    data = raw.copy()

print(f"Rows = {len(data)}")

# ============================================================================
# RAW COLUMNS -> WORKING COLUMNS (unit prefixes only)
# ============================================================================
data["Laa"] = data["L(Phase_A,Phase_A) [uH]"] * 1e-6
data["Lbb"] = data["L(Phase_B,Phase_B) [uH]"] * 1e-6
data["Lcc"] = data["L(Phase_C,Phase_C) [uH]"] * 1e-6
data["Lab"] = data["L(Phase_A,Phase_B) [nH]"] * 1e-9
data["Lac"] = data["L(Phase_A,Phase_C) [nH]"] * 1e-9
data["Lba"] = data["L(Phase_B,Phase_A) [nH]"] * 1e-9
data["Lbc"] = data["L(Phase_B,Phase_C) [nH]"] * 1e-9
data["Lca"] = data["L(Phase_C,Phase_A) [nH]"] * 1e-9
data["Lcb"] = data["L(Phase_C,Phase_B) [nH]"] * 1e-9

data["ia"] = data["InputCurrent(Phase_A) [mA]"] * 1e-3
data["ib"] = data["InputCurrent(Phase_B) [mA]"] * 1e-3
data["ic"] = data["InputCurrent(Phase_C) [mA]"] * 1e-3

data["fa"] = data["FluxLinkage(Phase_A) [Wb]"]
data["fb"] = data["FluxLinkage(Phase_B) [Wb]"]
data["fc"] = data["FluxLinkage(Phase_C) [Wb]"]

# ============================================================================
# ROTOR ELECTRICAL ANGLE + INTEGER-CYCLE ROW SELECTION
# ============================================================================
n_steps_col = int(raw["NStepsPerElectricalCycle []"].iloc[0])
block = n_steps_col + 1          # rows per operating point (verified = 25)

if rotor_mode == 'from_flux':
    nl = raw[raw["J [A_per_m2]"] == 0]
    if len(nl) == 0 or len(nl) % block != 0:
        raise ValueError(
            f"from_flux needs whole no-load blocks. Found {len(nl)} J=0 rows, "
            f"block size = {block}.")
    a0 = nl["FluxLinkage(Phase_A) [Wb]"].to_numpy()
    b0 = nl["FluxLinkage(Phase_B) [Wb]"].to_numpy()
    c0 = nl["FluxLinkage(Phase_C) [Wb]"].to_numpy()
    d0, q0, _ = park_vec(a0, b0, c0, 0.0)
    ang = np.unwrap(np.arctan2(q0, d0).reshape(-1, block), axis=1)
    theta_step = np.unwrap(np.angle(np.exp(1j * ang).mean(axis=0)))

    step_deg = np.rad2deg(np.median(np.diff(theta_step)))
    span_deg = np.rad2deg(theta_step[-1] - theta_step[0])
    n_per_cycle = int(round(360.0 / step_deg))
    n_keep = min(n_per_cycle * max(1, int(span_deg // 360)), block)
    print(f"Rotor sweep per block: {span_deg:.1f} elec deg in steps of "
          f"{step_deg:.2f} deg ({span_deg/360:.2f} cycles)")
    print(f"Averaging over first {n_keep} rows = "
          f"{n_keep*step_deg:.0f} deg (integer cycles)")

    if len(data) % block != 0:
        raise ValueError("Row count is not a whole number of blocks.")
    data["theta"] = np.tile(theta_step, len(data) // block)
    data["step_in_block"] = np.tile(np.arange(block), len(data) // block)
    data = data[data["step_in_block"] < n_keep].copy()

elif rotor_mode == 'sweep':
    remove_rows = int(np.floor(n_steps_col * init_period_to_remove / num_periods))
    kept = n_steps_col + 1 - remove_rows
    step_in_cycle = np.tile(np.arange(remove_rows, n_steps_col + 1),
                            len(data) // kept + 1)[:len(data)]
    data["theta"] = (np.deg2rad(rotor_ref_deg)
                     + step_in_cycle * np.deg2rad(elec_deg_per_cycle / n_steps_col))

else:  # 'fixed'
    data["theta"] = np.full(len(data), np.deg2rad(rotor_ref_deg))
    if auto_align:
        nl = data[data["J [A_per_m2]"] == 0]
        d0, q0, _ = park_vec(nl["fa"].to_numpy(), nl["fb"].to_numpy(),
                             nl["fc"].to_numpy(), nl["theta"].to_numpy())
        delta = np.arctan2(np.mean(q0), np.mean(d0))
        data["theta"] = data["theta"] + delta
        print(f"Auto-align: d-axis shifted by {np.rad2deg(delta):+.3f} deg")

# ============================================================================
# PER-ROW dq QUANTITIES
# ============================================================================
th = data["theta"].to_numpy()
d_i, q_i, _ = park_vec(data["ia"].to_numpy(), data["ib"].to_numpy(),
                       data["ic"].to_numpy(), th)
d_f, q_f, _ = park_vec(data["fa"].to_numpy(), data["fb"].to_numpy(),
                       data["fc"].to_numpy(), th)
data["id_raw"], data["iq_raw"] = d_i, q_i
data["psi_d_raw"], data["psi_q_raw"] = d_f, q_f

Ldm, Lqm = [], []
for _, row in data.iterrows():
    a, b = dq_inductances_matrix(row, row["theta"])
    Ldm.append(a); Lqm.append(b)
data["Ldm_raw"], data["Lqm_raw"] = Ldm, Lqm

# ============================================================================
# COLLAPSE EACH OPERATING POINT
# ============================================================================
m = data.groupby(["J [A_per_m2]", "TimingAdjustment [deg]"]).agg(
    id_raw=("id_raw", "mean"),
    iq_raw=("iq_raw", "mean"),
    psi_d_raw=("psi_d_raw", "mean"),
    psi_q_raw=("psi_q_raw", "mean"),
    Ldm_raw=("Ldm_raw", "mean"),
    Lqm_raw=("Lqm_raw", "mean"),
).reset_index()

# ============================================================================
# SCALE TO REAL MACHINE
# ============================================================================
N, s = turns_ratio, stack_factor
m["Id"]    = m["id_raw"]    / N
m["Iq"]    = m["iq_raw"]    / N
m["Psi_d"] = m["psi_d_raw"] * N * s
m["Psi_q"] = m["psi_q_raw"] * N * s
m["Ld_matrix"] = m["Ldm_raw"] * N**2 * s
m["Lq_matrix"] = m["Lqm_raw"] * N**2 * s
m["Psi_mag"] = np.hypot(m["Psi_d"], m["Psi_q"])

# ============================================================================
# PM FLUX
# ============================================================================
psi_pm = m.loc[m["J [A_per_m2]"] == 0, "Psi_d"].mean()
psi_q_nl = m.loc[m["J [A_per_m2]"] == 0, "Psi_q"].abs().mean()
print(f"\npsi_pm = {psi_pm:.6f} Wb   (no-load |psi_q| = {psi_q_nl:.2e} Wb, ~0 if aligned)")

# ---- Effective PM flux vs Iq (cross-saturation) ------------------------
# psi_pm_eff(Iq) := Psi_d along the Id=0 line. Q-axis current saturates iron
# shared with the magnet flux path, so the flux the magnets effectively link
# drops with |Iq|. Built from the near-zero-Id operating points and
# interpolated to every row's |Iq| (even function of Iq).
I_tot = np.hypot(m["Id"], m["Iq"])
on_axis = np.abs(m["Id"]) < np.maximum(0.5, 0.02 * I_tot)
line = (m.loc[on_axis, ["Iq", "Psi_d"]]
          .assign(aIq=lambda x: x["Iq"].abs().round(3))
          .groupby("aIq")["Psi_d"].mean().sort_index())
if len(line) < 3:
    raise ValueError("Not enough Id~0 points to build psi_pm_eff(Iq).")
m["Psi_pm_eff"] = np.interp(np.abs(m["Iq"]), line.index.to_numpy(),
                            line.to_numpy())
print(f"psi_pm_eff(Iq): {line.iloc[0]:.5f} Wb at Iq=0  ->  "
      f"{line.iloc[-1]:.5f} Wb at |Iq|={line.index[-1]:.0f} A "
      f"({100*(line.iloc[-1]-line.iloc[0])/line.iloc[0]:+.1f}% cross-saturation)")

# ============================================================================
# INCREMENTAL INDUCTANCES FROM THE FLUX MAP (finite differences on the
# polar J x timing grid, chain rule to d/dId, d/dIq). These are the
# trustworthy small-signal values.
# ============================================================================
Jv = np.sort(m["J [A_per_m2]"].unique())
Tv = np.sort(m["TimingAdjustment [deg]"].unique())

def pivot(col):
    return m.pivot(index="J [A_per_m2]", columns="TimingAdjustment [deg]",
                   values=col).loc[Jv, Tv].to_numpy()

Id_g, Iq_g = pivot("Id"), pivot("Iq")
Pd_g, Pq_g = pivot("Psi_d"), pivot("Psi_q")

dId_dJ, dId_dT = np.gradient(Id_g)
dIq_dJ, dIq_dT = np.gradient(Iq_g)
dPd_dJ, dPd_dT = np.gradient(Pd_g)
dPq_dJ, dPq_dT = np.gradient(Pq_g)

Ldd = np.full_like(Id_g, np.nan)   # dPsi_d/dId
Ldq = np.full_like(Id_g, np.nan)   # dPsi_d/dIq  (cross-saturation)
Lqd = np.full_like(Id_g, np.nan)   # dPsi_q/dId
Lqq = np.full_like(Id_g, np.nan)   # dPsi_q/dIq
for i in range(Id_g.shape[0]):
    for j in range(Id_g.shape[1]):
        Jac = np.array([[dId_dJ[i, j], dIq_dJ[i, j]],
                        [dId_dT[i, j], dIq_dT[i, j]]])
        if abs(np.linalg.det(Jac)) < 1e-12:
            continue                      # singular at J=0 (grid collapses)
        inv = np.linalg.inv(Jac)
        gd = inv @ np.array([dPd_dJ[i, j], dPd_dT[i, j]])
        gq = inv @ np.array([dPq_dJ[i, j], dPq_dT[i, j]])
        Ldd[i, j], Ldq[i, j] = gd
        Lqd[i, j], Lqq[i, j] = gq

flat = lambda A: pd.DataFrame(A, index=Jv, columns=Tv).stack()
inc = pd.concat([flat(Ldd), flat(Ldq), flat(Lqd), flat(Lqq)], axis=1)
inc.columns = ["Ld_inc", "Ldq_inc", "Lqd_inc", "Lq_inc"]
inc.index.names = ["J [A_per_m2]", "TimingAdjustment [deg]"]
m = m.merge(inc.reset_index(), on=["J [A_per_m2]", "TimingAdjustment [deg]"])

# ============================================================================
# INTERNAL ONLY — APPARENT (SECANT) INDUCTANCES: NOT EXPORTED
# ============================================================================
# What: Ld_app = (Psi_d - psi_pm_eff(Iq)) / Id and Lq_app = Psi_q / Iq — the
# inductances defined directly from this file's flux linkages, i.e. the
# values for which Psi = L*I exactly reproduces the FEA flux at each point.
# Why computed: the data audit found the FEA inductance matrix inconsistent
# with the flux linkage columns (Maxwell computes the matrix by unit-current
# perturbation with PM effects excluded); these were built as the
# flux-consistent alternative, and they are still used internally below to
# validate/fill and for the console comparison.
# Why not exported: they show strong load dependence and near-unity/inverted
# saliency at operating state, which does not match the motor's design intent
# or the locked-rotor test results (saliency ~1.06, Lq > Ld). Team decision
# (July 2026): export the matrix values instead (renamed Ld/Lq below).
ok_d = (np.abs(m["Id"]) > 0.02 * I_tot) & (np.abs(m["Id"]) > 0.5)
ok_q = (np.abs(m["Iq"]) > 0.02 * I_tot) & (np.abs(m["Iq"]) > 0.5)
m["Ld_app"] = np.where(ok_d, (m["Psi_d"] - m["Psi_pm_eff"]) / m["Id"], np.nan)
m["Lq_app"] = np.where(ok_q, m["Psi_q"] / m["Iq"], np.nan)

# Fill undefined 0/0 points with their mathematical limits (the incremental
# values), so the internal grid is complete.
m["Ld_app"] = m["Ld_app"].fillna(m["Ld_inc"])
m["Lq_app"] = m["Lq_app"].fillna(m["Lq_inc"])
pure_d = m.loc[m["Ld_inc"].notna() & (np.abs(m["Iq"]) < 0.5), "Ld_inc"]
pure_q = m.loc[m["Lq_inc"].notna() & (np.abs(m["Id"]) < 0.5), "Lq_inc"]
m["Ld_app"] = m["Ld_app"].fillna(pure_d.iloc[0] if len(pure_d) else np.nan)
m["Lq_app"] = m["Lq_app"].fillna(pure_q.iloc[0] if len(pure_q) else np.nan)

# Saliency ratio from the EXPORTED (matrix) inductances: Lq/Ld > 1
# everywhere (~1.10), consistent with the IPM design intent and the
# locked-rotor measurement (~1.06). The flux-derived saliency
# (Lq_app/Ld_app, near or below 1 at operating state) is internal only.
m["Saliency"] = m["Lq_matrix"] / m["Ld_matrix"]

# ============================================================================
# TORQUE (exact from the flux map)
# ============================================================================
m["Torque_Nm"] = 1.5 * pole_pairs * (m["Psi_pm_eff"] + (m["Ld_matrix"]- m["Lq_matrix"])* m["Id"]) * m["Iq"]

# ============================================================================
# DIAGNOSTICS: console comparison of the inductance definitions (all three
# stay visible here even though only the matrix values are exported).
# ============================================================================
print("\n--- Inductance definitions compared (uH, real machine) ---")
print("matrix = Maxwell perturbation values (EXPORTED as Ld/Lq, per team decision)")
print("incremental/apparent = flux-linkage-derived (internal only; see notes)")
sub = m.dropna(subset=["Lq_inc"])
print(f"Lq: incremental {sub.Lq_inc.min()*1e6:6.0f}..{sub.Lq_inc.max()*1e6:4.0f} | "
      f"apparent {np.nanmin(m.Lq_app)*1e6:4.0f}..{np.nanmax(m.Lq_app)*1e6:4.0f} | "
      f"matrix {m.Lq_matrix.min()*1e6:4.0f}..{m.Lq_matrix.max()*1e6:4.0f}")
print(f"Ld: incremental {sub.Ld_inc.min()*1e6:6.0f}..{sub.Ld_inc.max()*1e6:4.0f} | "
      f"apparent {np.nanmin(m.Ld_app)*1e6:4.0f}..{np.nanmax(m.Ld_app)*1e6:4.0f} | "
      f"matrix {m.Ld_matrix.min()*1e6:4.0f}..{m.Ld_matrix.max()*1e6:4.0f}")
print(f"cross-coupling Ldq_inc: {np.nanmin(m.Ldq_inc)*1e6:.0f}..{np.nanmax(m.Ldq_inc)*1e6:.0f} uH (internal)")

m["Imag"] = np.hypot(m.Id, m.Iq)
m["beta"] = np.degrees(np.arctan2(-m.Id, m.Iq))
peak = m.loc[m["Torque_Nm"].idxmax()]
print(f"\nPeak torque {peak.Torque_Nm:.2f} Nm at |I|={peak.Imag:.0f} A, "
      f"beta={peak.beta:.0f} deg (MTPA angle at max current)")

# ============================================================================
# EXPORT — Ld/Lq columns are the Maxwell matrix values (renamed on export,
# per team decision; provenance recorded in the module docstring).
# ============================================================================
out = m[[
    "Id", "Iq",
    "Psi_d", "Psi_q", "Psi_mag",
    "Ld_matrix", "Lq_matrix",
    "Saliency",
    "Torque_Nm",
]].copy()
out = out.rename(columns={"Ld_matrix": "Ld", "Lq_matrix": "Lq"})
out.insert(0, "Psi_pm_eff", m["Psi_pm_eff"].to_numpy())
for c in ["Ld", "Lq"]:
    out[c + "_uH"] = out[c] * 1e6
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
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
panels = [("Ld_uH", "Ld [uH]"),
          ("Lq_uH", "Lq [uH]"),
          ("Saliency", "Saliency Lq/Ld [-]"),
          ("Psi_mag", "|psi| [Wb]"),
          ("Torque_Nm", "Torque [Nm]"),
          ("Psi_pm_eff", "psi_pm_eff [Wb]")]
for a, (col, lab) in zip(ax.flat, panels):
    v = out[col]
    sc = a.scatter(out["Id"], out["Iq"], c=v)
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
