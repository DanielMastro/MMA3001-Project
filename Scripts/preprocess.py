"""
preprocess.py - winding table -> flux map.  Refactored from History/preprocess_v3.py, the original inductance script.

Where this fits in the project
------------------------------
This is the first stage of the pipeline. It reads one FEA winding table in the
Ansys Maxwell export format and returns one row per operating point holding
Id, Iq, psi_d, psi_q, the inductances and torque. The two inputs are:

    Input_Data/Cobalt/cobalt_input_data.csv          the team's Maxwell 2D FEA
    Input_Data/Zenodo_IPM/zenodo_input_data.csv      open IPM data, converted to
                                                     this format by
                                                     create_zenodo_input_data.py

models.py then fits flux models to this table, and evaluate.py checks them.

The steps (same methods and results as preprocess_v3)
---------------------------------------------
  1. load_winding_table()  read the CSV and check it is usable
  2. rotor_angle()         rotor electrical angle at every step, read from the
                           direction of the no-load (J = 0) flux
  3. dq_per_row()          Park transform of currents, flux and the 3x3
                           inductance matrix at that rotor angle
  4. average_cycles()      average each operating point over a whole number of
                           electrical cycles (numerical integration)
  5. scale()               2-turn 2D model -> real machine (turns ratio, stack factor)
  6. pm_flux()             psi_pm and the cross-saturated psi_pm_eff(Iq)
  7. grid_inductances()    incremental inductances by finite differences on the
                           FEA grid, and apparent inductances, as in preprocess_v3
  build_flux_map() runs all seven and adds torque.

What changed from preprocess_v3, and why
--------------------------------
  * The code is split into functions, and the settings (file, turns ratio,
    stack factor, pole pairs) are arguments rather than constants at the top.
    That lets the same code run on both motors, be tested with pytest and be
    documented with pdoc. Cobalt: turns_ratio=33/2, stack_factor=0.9,
    pole_pairs=5. Zenodo: 1, 1, 8.
  * The Park transform of the inductance matrix is done for all rows at once
    (vectorised). preprocess_v3's row-by-row iterrows loop is kept, selectable with
    inductance_method="loop", so evaluate.py can time the two against each
    other. Both give the same numbers.
  * Steps per electrical cycle use the MEAN rotor-angle step instead of the
    median. Cobalt still gets 16 steps. Zenodo's 60 small steps carry slot
    ripple, which can push the median far enough to round to 59.
  * The incremental-inductance table is assembled without DataFrame.stack(),
    and merged with how="left". The results are the same, but the J = 0 rows
    are kept on every pandas version.
  * Torque now uses the flux formula (derive.torque_from_flux). preprocess_v3's
    matrix-inductance formula is kept as the column Torque_matrix_Nm for
    comparison only (evaluate.torque_formula_comparison).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from derive import matrix_torque, torque_from_flux

# ============================================================================
# COLUMN NAMES (the Maxwell winding-table export; both input files use these)
# ============================================================================
COL_J = "J [A_per_m2]"
COL_NSTEPS = "NStepsPerElectricalCycle []"
COL_TIMING = "TimingAdjustment [deg]"

REQUIRED_COLUMNS = [
    COL_J, COL_NSTEPS, "RPM [rpm]", COL_TIMING, "Time [s]",
    "L(Phase_A,Phase_A) [uH]", "L(Phase_B,Phase_A) [nH]", "L(Phase_C,Phase_A) [nH]",
    "L(Phase_A,Phase_B) [nH]", "L(Phase_B,Phase_B) [uH]", "L(Phase_C,Phase_B) [nH]",
    "L(Phase_A,Phase_C) [nH]", "L(Phase_B,Phase_C) [nH]", "L(Phase_C,Phase_C) [uH]",
    "FluxLinkage(Phase_A) [Wb]", "FluxLinkage(Phase_B) [Wb]", "FluxLinkage(Phase_C) [Wb]",
    "InducedVoltage(Phase_A) [uv]", "InducedVoltage(Phase_B) [uv]", "InducedVoltage(Phase_C) [uv]",
    "InputCurrent(Phase_A) [mA]", "InputCurrent(Phase_B) [mA]", "InputCurrent(Phase_C) [mA]",
]
INDUCTANCE_COLUMNS = [c for c in REQUIRED_COLUMNS if c.startswith("L(")]


# ============================================================================
# PARK / CLARKE (amplitude-invariant, 2/3 convention) - unchanged from preprocess_v3
# ============================================================================
def park_vec(a, b, c, theta):
    """abc -> dq0 for a vector quantity (current, flux, voltage) at rotor
    electrical angle theta [rad]. Amplitude-invariant: a balanced set of peak
    amplitude X gives |dq| = X."""
    d = (2/3) * (a*np.cos(theta) + b*np.cos(theta - 2*np.pi/3) + c*np.cos(theta + 2*np.pi/3))
    q = -(2/3) * (a*np.sin(theta) + b*np.sin(theta - 2*np.pi/3) + c*np.sin(theta + 2*np.pi/3))
    z = (1/3) * (a + b + c)
    return d, q, z


def park_matrix(theta):
    """The 3x3 Park matrix T(theta), so that [d, q, 0] = T @ [a, b, c]."""
    return (2/3) * np.array([
        [np.cos(theta), np.cos(theta - 2*np.pi/3), np.cos(theta + 2*np.pi/3)],
        [-np.sin(theta), -np.sin(theta - 2*np.pi/3), -np.sin(theta + 2*np.pi/3)],
        [0.5, 0.5, 0.5],
    ])


def dq_inductances_matrix(row, theta):
    """dq transform of the FEA's abc inductance matrix for ONE row:
    L_dq0 = T @ L_abc @ inv(T). Returns (Ld, Lq). This is preprocess_v3's per-row
    version, used by inductance_method="loop"."""
    T = park_matrix(theta)
    L = np.array([
        [row["Laa"], row["Lab"], row["Lac"]],
        [row["Lba"], row["Lbb"], row["Lbc"]],
        [row["Lca"], row["Lcb"], row["Lcc"]],
    ])
    D = T @ L @ np.linalg.inv(T)
    return D[0, 0], D[1, 1]


def park_matrix_stack(theta):
    """park_matrix() for many angles at once: returns an (n, 3, 3) array."""
    theta = np.asarray(theta, dtype=float)
    T = np.empty((theta.size, 3, 3))
    T[:, 0, 0] = np.cos(theta)
    T[:, 0, 1] = np.cos(theta - 2*np.pi/3)
    T[:, 0, 2] = np.cos(theta + 2*np.pi/3)
    T[:, 1, 0] = -np.sin(theta)
    T[:, 1, 1] = -np.sin(theta - 2*np.pi/3)
    T[:, 1, 2] = -np.sin(theta + 2*np.pi/3)
    T[:, 2, :] = 0.5
    return (2/3) * T


# ============================================================================
# 1. LOAD AND CHECK
# ============================================================================
def load_winding_table(path):
    """Read a winding-table CSV and check it can be processed.

    Raises a clear error if:
      * the file is missing, or a required column is missing
      * any current, flux or setting column contains NaN
      * the row count is not a whole number of operating-point blocks
        (each block = NStepsPerElectricalCycle + 1 rows)
      * there are no no-load (J = 0) rows, which the rotor angle is read from
      * the phase currents do not sum to zero (an unbalanced or mislabelled set)

    NaN in the inductance-matrix columns only gives a warning: those columns
    are optional, and their dq values will be NaN.
    Returns the raw DataFrame, units unchanged.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Winding table not found: {path}")
    raw = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    core = [c for c in REQUIRED_COLUMNS if c not in INDUCTANCE_COLUMNS]
    if raw[core].isna().to_numpy().any():
        raise ValueError(f"{path.name} contains NaN values in: "
                         f"{[c for c in core if raw[c].isna().any()]}")
    if raw[INDUCTANCE_COLUMNS].isna().to_numpy().any():
        print(f"Warning: {path.name} has NaN inductance-matrix values; "
              "matrix Ld/Lq will be NaN (flux-derived inductances are unaffected).")

    block = int(raw[COL_NSTEPS].iloc[0]) + 1
    if len(raw) % block != 0:
        raise ValueError(f"{path.name}: {len(raw)} rows is not a whole number of "
                         f"{block}-row operating-point blocks.")
    if not (raw[COL_J] == 0).any():
        raise ValueError(f"{path.name} has no no-load (J = 0) rows; "
                         "the rotor angle cannot be found.")

    i_sum = (raw["InputCurrent(Phase_A) [mA]"] + raw["InputCurrent(Phase_B) [mA]"]
             + raw["InputCurrent(Phase_C) [mA]"]).abs().max()
    i_max = raw["InputCurrent(Phase_A) [mA]"].abs().max()
    if i_max > 0 and i_sum > 1e-3 * i_max:
        raise ValueError(f"{path.name}: phase currents do not sum to zero "
                         f"(max |ia+ib+ic| = {i_sum:.3g} mA).")
    return raw


def working_columns(data):
    """RAW COLUMNS -> WORKING COLUMNS (unit prefixes only), as in preprocess_v3.
    Inductances uH/nH -> H, currents mA -> A, flux already in Wb."""
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
    return data


# ============================================================================
# 2. ROTOR ELECTRICAL ANGLE + INTEGER-CYCLE ROW SELECTION
# ============================================================================
def rotor_angle(raw, rotor_mode="from_flux", rotor_ref_deg=0.0, auto_align=True,
                init_period_to_remove=0.5, num_periods=1.5, elec_deg_per_cycle=360.0,
                verbose=True):
    """Add the rotor electrical angle "theta" [rad] to every row (preprocess_v3 logic).

    rotor_mode:
      "from_flux" (default) - at no load the flux points along the d-axis, so
          the rotor angle at each step IS the angle of the no-load flux. It is
          read from the J = 0 blocks and averaged across them (circular mean).
          Only the first n_keep rows of each block are kept, a whole number of
          electrical cycles, so the averaging in step 4 is not biased. Cobalt
          stores 1.5 cycles in 25 rows and keeps 16; Zenodo stores 1 cycle in
          61 rows and keeps 60.
      "fixed" - one constant angle, optionally auto-aligned so no-load psi_q = 0.
      "sweep" - legacy: angle rebuilt from the step index (needs
          init_period_to_remove, num_periods and elec_deg_per_cycle to be right).

    Returns the working DataFrame (rows kept + working columns + "theta").
    """
    n_steps_col = int(raw[COL_NSTEPS].iloc[0])
    block = n_steps_col + 1          # rows per operating point (25 Cobalt, 61 Zenodo)

    # ---- which rows to keep (preprocess_v3 "LOAD" section) ----
    if rotor_mode == "sweep":
        remove_rows = int(np.floor(n_steps_col * init_period_to_remove / num_periods))
        idx = (np.arange(len(raw))
               .reshape(-1, block)[:, remove_rows:]
               .flatten())
        data = raw.iloc[idx].copy()
    else:
        data = raw.copy()
    data = working_columns(data)

    if rotor_mode == "from_flux":
        nl = raw[raw[COL_J] == 0]
        if len(nl) == 0 or len(nl) % block != 0:
            raise ValueError(
                f"from_flux needs whole no-load blocks. Found {len(nl)} J=0 rows, "
                f"block size = {block}.")
        a0 = nl["FluxLinkage(Phase_A) [Wb]"].to_numpy()
        b0 = nl["FluxLinkage(Phase_B) [Wb]"].to_numpy()
        c0 = nl["FluxLinkage(Phase_C) [Wb]"].to_numpy()
        d0, q0, _ = park_vec(a0, b0, c0, 0.0)               # theta = 0 -> stationary alpha, beta
        ang = np.unwrap(np.arctan2(q0, d0).reshape(-1, block), axis=1)
        theta_step = np.unwrap(np.angle(np.exp(1j * ang).mean(axis=0)))   # circular mean per step

        # Mean step (preprocess_v3 used the median; see the module docstring)
        step_deg = np.rad2deg(np.mean(np.diff(theta_step)))
        span_deg = np.rad2deg(theta_step[-1] - theta_step[0])
        n_per_cycle = int(round(360.0 / step_deg))
        n_keep = min(n_per_cycle * max(1, int(round(span_deg, 6) // 360)), block)
        if verbose:
            print(f"Rotor sweep per block: {span_deg:.1f} elec deg in steps of "
                  f"{step_deg:.2f} deg ({span_deg/360:.2f} cycles)")
            print(f"Averaging over first {n_keep} rows = "
                  f"{n_keep*step_deg:.0f} deg (integer cycles)")

        if len(data) % block != 0:
            raise ValueError("Row count is not a whole number of blocks.")
        data["theta"] = np.tile(theta_step, len(data) // block)
        data["step_in_block"] = np.tile(np.arange(block), len(data) // block)
        data = data[data["step_in_block"] < n_keep].copy()

    elif rotor_mode == "sweep":
        remove_rows = int(np.floor(n_steps_col * init_period_to_remove / num_periods))
        kept = n_steps_col + 1 - remove_rows
        step_in_cycle = np.tile(np.arange(remove_rows, n_steps_col + 1),
                                len(data) // kept + 1)[:len(data)]
        data["theta"] = (np.deg2rad(rotor_ref_deg)
                         + step_in_cycle * np.deg2rad(elec_deg_per_cycle / n_steps_col))

    elif rotor_mode == "fixed":
        data["theta"] = np.full(len(data), np.deg2rad(rotor_ref_deg))
        if auto_align:
            nl = data[data[COL_J] == 0]
            d0, q0, _ = park_vec(nl["fa"].to_numpy(), nl["fb"].to_numpy(),
                                 nl["fc"].to_numpy(), nl["theta"].to_numpy())
            delta = np.arctan2(np.mean(q0), np.mean(d0))
            data["theta"] = data["theta"] + delta
            if verbose:
                print(f"Auto-align: d-axis shifted by {np.rad2deg(delta):+.3f} deg")
    else:
        raise ValueError("rotor_mode must be 'from_flux', 'fixed' or 'sweep'")
    return data


# ============================================================================
# 3. PER-ROW dq QUANTITIES
# ============================================================================
def dq_per_row(data, inductance_method="vectorised"):
    """Park-transform every row at its rotor angle.

    Adds id_raw, iq_raw, psi_d_raw, psi_q_raw (model turns, before scaling) and
    Ldm_raw, Lqm_raw (the dq transform of the Maxwell inductance matrix).

    inductance_method:
      "vectorised" - T @ L @ inv(T) for every row at once (default)
      "loop"       - preprocess_v3's row-by-row iterrows loop (slow; kept for timing)
    Both compute exactly the same matrix product.
    """
    th = data["theta"].to_numpy()
    d_i, q_i, _ = park_vec(data["ia"].to_numpy(), data["ib"].to_numpy(),
                           data["ic"].to_numpy(), th)
    d_f, q_f, _ = park_vec(data["fa"].to_numpy(), data["fb"].to_numpy(),
                           data["fc"].to_numpy(), th)
    data["id_raw"], data["iq_raw"] = d_i, q_i
    data["psi_d_raw"], data["psi_q_raw"] = d_f, q_f

    if inductance_method == "loop":
        # preprocess_v3, unchanged: one 3x3 matrix product per row
        Ldm, Lqm = [], []
        for _, row in data.iterrows():
            a, b = dq_inductances_matrix(row, row["theta"])
            Ldm.append(a); Lqm.append(b)
    elif inductance_method == "vectorised":
        # Stack every row's 3x3 abc matrix into one (n, 3, 3) array
        L = np.empty((len(data), 3, 3))
        L[:, 0, 0], L[:, 0, 1], L[:, 0, 2] = data["Laa"], data["Lab"], data["Lac"]
        L[:, 1, 0], L[:, 1, 1], L[:, 1, 2] = data["Lba"], data["Lbb"], data["Lbc"]
        L[:, 2, 0], L[:, 2, 1], L[:, 2, 2] = data["Lca"], data["Lcb"], data["Lcc"]
        T = park_matrix_stack(th)
        D = T @ L @ np.linalg.inv(T)            # same product as preprocess_v3, all rows at once
        Ldm, Lqm = D[:, 0, 0], D[:, 1, 1]
    else:
        raise ValueError("inductance_method must be 'vectorised' or 'loop'")
    data["Ldm_raw"], data["Lqm_raw"] = Ldm, Lqm
    return data


# ============================================================================
# 4. COLLAPSE EACH OPERATING POINT (average over whole electrical cycles)
# ============================================================================
def average_cycles(data):
    """One row per operating point (J, TimingAdjustment): the mean of every
    kept row. Because rotor_angle() kept a whole number of cycles, this mean is
    the cycle average, i.e. the integral over a period divided by the period."""
    return data.groupby([COL_J, COL_TIMING]).agg(
        id_raw=("id_raw", "mean"),
        iq_raw=("iq_raw", "mean"),
        psi_d_raw=("psi_d_raw", "mean"),
        psi_q_raw=("psi_q_raw", "mean"),
        Ldm_raw=("Ldm_raw", "mean"),
        Lqm_raw=("Lqm_raw", "mean"),
    ).reset_index()


# ============================================================================
# 5. SCALE TO REAL MACHINE
# ============================================================================
def scale(m, turns_ratio, stack_factor):
    """Model turns -> real machine:
    flux ~ N*s, current ~ 1/N, inductance ~ N^2*s (N = turns ratio, s = stack factor)."""
    N, s = turns_ratio, stack_factor
    m["Id"] = m["id_raw"] / N
    m["Iq"] = m["iq_raw"] / N
    m["Psi_d"] = m["psi_d_raw"] * N * s
    m["Psi_q"] = m["psi_q_raw"] * N * s
    m["Ld_matrix"] = m["Ldm_raw"] * N**2 * s
    m["Lq_matrix"] = m["Lqm_raw"] * N**2 * s
    m["Psi_mag"] = np.hypot(m["Psi_d"], m["Psi_q"])
    return m


# ============================================================================
# 6. PM FLUX
# ============================================================================
def pm_flux(m, verbose=True):
    """psi_pm (no-load psi_d) and psi_pm_eff(Iq), as in preprocess_v3.

    psi_pm_eff(Iq) := psi_d along the Id = 0 line. Q-axis current saturates iron
    shared with the magnet flux path, so the flux the magnets effectively link
    drops with |Iq|. It is built from the near-zero-Id operating points and
    interpolated to every row's |Iq| (an even function of Iq).
    Returns (m, psi_pm).
    """
    psi_pm = m.loc[m[COL_J] == 0, "Psi_d"].mean()
    psi_q_nl = m.loc[m[COL_J] == 0, "Psi_q"].abs().mean()
    if verbose:
        print(f"\npsi_pm = {psi_pm:.6f} Wb   (no-load |psi_q| = {psi_q_nl:.2e} Wb, ~0 if aligned)")

    I_tot = np.hypot(m["Id"], m["Iq"])
    on_axis = np.abs(m["Id"]) < np.maximum(0.5, 0.02 * I_tot)
    line = (m.loc[on_axis, ["Iq", "Psi_d"]]
              .assign(aIq=lambda x: x["Iq"].abs().round(3))
              .groupby("aIq")["Psi_d"].mean().sort_index())
    if len(line) < 3:
        raise ValueError("Not enough Id~0 points to build psi_pm_eff(Iq).")
    m["Psi_pm_eff"] = np.interp(np.abs(m["Iq"]), line.index.to_numpy(), line.to_numpy())
    if verbose:
        print(f"psi_pm_eff(Iq): {line.iloc[0]:.5f} Wb at Iq=0  ->  "
              f"{line.iloc[-1]:.5f} Wb at |Iq|={line.index[-1]:.0f} A "
              f"({100*(line.iloc[-1]-line.iloc[0])/line.iloc[0]:+.1f}% cross-saturation)")
    return m, psi_pm


# ============================================================================
# 7. INDUCTANCES ON THE FEA GRID (preprocess_v3's incremental + apparent sections)
# ============================================================================
def grid_inductances(m):
    """Incremental and apparent inductances directly from the FEA points.

    Incremental: finite differences (np.gradient, 3-point central) on the polar
    J x TimingAdjustment grid, then the chain rule (a 2x2 Jacobian solve) to get
    d/dId and d/dIq. Columns Ld_inc, Ldq_inc, Lqd_inc, Lq_inc. They are NaN at
    J = 0, where the polar grid collapses to a point.

    Apparent: Ld_app = (psi_d - psi_pm_eff)/Id, Lq_app = psi_q/Iq. The 0/0
    points are filled with their limits, the incremental values.

    Also Saliency = Lq_matrix / Ld_matrix (preprocess_v3's exported saliency).
    """
    Jv = np.sort(m[COL_J].unique())
    Tv = np.sort(m[COL_TIMING].unique())

    def pivot(col):
        return m.pivot(index=COL_J, columns=COL_TIMING, values=col).loc[Jv, Tv].to_numpy()

    Id_g, Iq_g = pivot("Id"), pivot("Iq")
    Pd_g, Pq_g = pivot("Psi_d"), pivot("Psi_q")

    # Derivatives along the two grid directions (current magnitude J, angle T)
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
            # Chain rule: [dpsi/dJ, dpsi/dT] = Jac @ [dpsi/dId, dpsi/dIq]
            Jac = np.array([[dId_dJ[i, j], dIq_dJ[i, j]],
                            [dId_dT[i, j], dIq_dT[i, j]]])
            if abs(np.linalg.det(Jac)) < 1e-12:
                continue                      # singular at J=0 (grid collapses)
            inv = np.linalg.inv(Jac)
            gd = inv @ np.array([dPd_dJ[i, j], dPd_dT[i, j]])
            gq = inv @ np.array([dPq_dJ[i, j], dPq_dT[i, j]])
            Ldd[i, j], Ldq[i, j] = gd
            Lqd[i, j], Lqq[i, j] = gq

    # Back to one row per operating point (the same as preprocess_v3's .stack(), but
    # without dropping the NaN J=0 rows on older pandas versions)
    JJ, TT = np.meshgrid(Jv, Tv, indexing="ij")
    inc = pd.DataFrame({COL_J: JJ.ravel(), COL_TIMING: TT.ravel(),
                        "Ld_inc": Ldd.ravel(), "Ldq_inc": Ldq.ravel(),
                        "Lqd_inc": Lqd.ravel(), "Lq_inc": Lqq.ravel()})
    m = m.merge(inc, on=[COL_J, COL_TIMING], how="left")

    # ---- apparent (secant) inductances ----
    I_tot = np.hypot(m["Id"], m["Iq"])
    ok_d = (np.abs(m["Id"]) > 0.02 * I_tot) & (np.abs(m["Id"]) > 0.5)
    ok_q = (np.abs(m["Iq"]) > 0.02 * I_tot) & (np.abs(m["Iq"]) > 0.5)
    m["Ld_app"] = np.where(ok_d, (m["Psi_d"] - m["Psi_pm_eff"]) / m["Id"], np.nan)
    m["Lq_app"] = np.where(ok_q, m["Psi_q"] / m["Iq"], np.nan)

    # Fill undefined 0/0 points with their mathematical limits (the incremental
    # values), so the grid is complete
    m["Ld_app"] = m["Ld_app"].fillna(m["Ld_inc"])
    m["Lq_app"] = m["Lq_app"].fillna(m["Lq_inc"])
    pure_d = m.loc[m["Ld_inc"].notna() & (np.abs(m["Iq"]) < 0.5), "Ld_inc"]
    pure_q = m.loc[m["Lq_inc"].notna() & (np.abs(m["Id"]) < 0.5), "Lq_inc"]
    m["Ld_app"] = m["Ld_app"].fillna(pure_d.iloc[0] if len(pure_d) else np.nan)
    m["Lq_app"] = m["Lq_app"].fillna(pure_q.iloc[0] if len(pure_q) else np.nan)

    # Saliency from the matrix inductances, as preprocess_v3 exported it
    m["Saliency"] = m["Lq_matrix"] / m["Ld_matrix"]
    return m


# ============================================================================
# FEA TORQUE (only the Zenodo data has it)
# ============================================================================
def attach_fea_torque(m, operating_points_csv):
    """Add the FEA's own torque, column Torque_FEA_Nm, matched on (J, TimingAdjustment).

    The Zenodo FEA computed torque directly. create_zenodo_input_data.py saves
    it, averaged per operating point, in zenodo_input_data_operating_points.csv.
    It is the independent check for torque_from_flux().
    """
    op = pd.read_csv(operating_points_csv)
    key = lambda df: pd.DataFrame({"_J": df[COL_J].round(6), "_T": df[COL_TIMING].round(6)})
    left = pd.concat([m, key(m)], axis=1)
    right = pd.concat([key(op), op[["Torque_FEA_Nm"]]], axis=1).drop_duplicates(["_J", "_T"])
    out = left.merge(right, on=["_J", "_T"], how="left").drop(columns=["_J", "_T"])
    if out["Torque_FEA_Nm"].isna().any():
        print("Warning: some operating points have no FEA torque match.")
    return out


# ============================================================================
# THE WHOLE STAGE
# ============================================================================
def build_flux_map(path, turns_ratio, stack_factor, pole_pairs, rotor_mode="from_flux",
                   inductance_method="vectorised", fea_torque_csv=None, verbose=True,
                   **rotor_kwargs):
    """Run steps 1-7 on one winding table and return the flux map.

    One row per FEA operating point, with columns:
      J [A_per_m2], TimingAdjustment [deg]  the FEA sweep settings
      Id, Iq [A]                            real-machine dq current (peak)
      Psi_d, Psi_q, Psi_mag, Psi_pm_eff, Psi_pm [Wb]
      Ld_matrix, Lq_matrix [H]              Maxwell's matrix, dq-transformed
      Ld_inc, Ldq_inc, Lqd_inc, Lq_inc [H]  finite differences on the FEA grid
      Ld_app, Lq_app [H]                    apparent (secant)
      Saliency                              Lq_matrix / Ld_matrix
      Torque_Nm                             flux formula, 1.5 p (psi_d Iq - psi_q Id)
      Torque_matrix_Nm                      preprocess_v3's matrix formula (comparison)
      Imag [A], beta [deg]                  current magnitude and angle
      Torque_FEA_Nm                         only if fea_torque_csv is given
    """
    raw = load_winding_table(path)                                   # 1
    data = rotor_angle(raw, rotor_mode, verbose=verbose, **rotor_kwargs)   # 2
    data = dq_per_row(data, inductance_method)                       # 3
    m = average_cycles(data)                                         # 4
    m = scale(m, turns_ratio, stack_factor)                          # 5
    m, psi_pm = pm_flux(m, verbose)                                  # 6
    m = grid_inductances(m)                                          # 7

    # Torque: the flux formula is now the main one. preprocess_v3's formula is kept
    # beside it so the two can be compared (report section 4).
    m["Torque_Nm"] = torque_from_flux(m["Psi_d"], m["Psi_q"], m["Id"], m["Iq"], pole_pairs)
    m["Torque_matrix_Nm"] = matrix_torque(m["Psi_pm_eff"], m["Ld_matrix"], m["Lq_matrix"],
                                          m["Id"], m["Iq"], pole_pairs)
    m["Psi_pm"] = psi_pm
    m["Imag"] = np.hypot(m["Id"], m["Iq"])
    m["beta"] = np.degrees(np.arctan2(-m["Id"], m["Iq"]))
    if fea_torque_csv is not None:
        m = attach_fea_torque(m, fea_torque_csv)
    if verbose:
        print_diagnostics(m)
    return m


def print_diagnostics(m):
    """preprocess_v3's console comparison of the inductance definitions, plus peak torque."""
    print("\n--- Inductance definitions compared (uH, real machine) ---")
    sub = m.dropna(subset=["Lq_inc"])
    print(f"Lq: incremental {sub.Lq_inc.min()*1e6:6.0f}..{sub.Lq_inc.max()*1e6:4.0f} | "
          f"apparent {np.nanmin(m.Lq_app)*1e6:4.0f}..{np.nanmax(m.Lq_app)*1e6:4.0f} | "
          f"matrix {m.Lq_matrix.min()*1e6:4.0f}..{m.Lq_matrix.max()*1e6:4.0f}")
    print(f"Ld: incremental {sub.Ld_inc.min()*1e6:6.0f}..{sub.Ld_inc.max()*1e6:4.0f} | "
          f"apparent {np.nanmin(m.Ld_app)*1e6:4.0f}..{np.nanmax(m.Ld_app)*1e6:4.0f} | "
          f"matrix {m.Ld_matrix.min()*1e6:4.0f}..{m.Ld_matrix.max()*1e6:4.0f}")
    print(f"cross-coupling Ldq_inc: {np.nanmin(m.Ldq_inc)*1e6:.0f}..{np.nanmax(m.Ldq_inc)*1e6:.0f} uH")
    peak = m.loc[m["Torque_Nm"].idxmax()]
    print(f"\nPeak torque (flux formula) {peak.Torque_Nm:.2f} Nm at |I|={peak.Imag:.0f} A, "
          f"beta={peak.beta:.0f} deg   [preprocess_v3 matrix formula peak: {m.Torque_matrix_Nm.max():.2f} Nm]")


# ============================================================================
# HELPERS FOR THE LATER STAGES
# ============================================================================
def unique_operating_points(m):
    """Drop repeated physical operating points before fitting a model.

    The FEA sweeps repeat some points: J = 0 is run once per angle, and on a
    full-circle sweep TimingAdjustment = -180 and +180 are the same angle. A
    repeat in both the training and test folds would make cross-validation look
    better than it is, so only the first copy is kept.
    """
    t = ((m[COL_TIMING] + 180.0) % 360.0) - 180.0               # -180 and +180 -> same
    t = np.where(m[COL_J] == 0, 0.0, t)                         # every J=0 row is one point
    key = pd.DataFrame({"J": m[COL_J].round(9).to_numpy(), "T": np.round(t, 6)})
    return m.loc[~key.duplicated().to_numpy()].reset_index(drop=True)


def base_values(m, pole_pairs):
    """Per-unit bases for reporting errors on a common scale across motors:
    I_base = largest |I|, psi_base = psi_pm, T_base = 1.5 * p * psi_pm * I_base."""
    I_base = float(np.hypot(m["Id"], m["Iq"]).max())
    psi_base = float(m["Psi_pm"].iloc[0])
    return {"I_base": I_base, "psi_base": psi_base,
            "T_base": 1.5 * pole_pairs * psi_base * I_base}


def plot_flux_map(m, title="", path=None, show=False):
    """preprocess_v3's 2x3 scatter of the flux map (torque panel now uses the flux formula).
    Saved to `path` if given; shown on screen only if show=True."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")                   # draw off-screen (no window)
    import matplotlib.pyplot as plt

    panels = [(m["Ld_matrix"] * 1e6, "Ld matrix [uH]"),
              (m["Lq_matrix"] * 1e6, "Lq matrix [uH]"),
              (m["Saliency"], "Saliency Lq/Ld [-]"),
              (m["Psi_mag"], "|psi| [Wb]"),
              (m["Torque_Nm"], "Torque [Nm]"),
              (m["Psi_pm_eff"], "psi_pm_eff [Wb]")]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for a, (v, lab) in zip(ax.flat, panels):
        sc = a.scatter(m["Id"], m["Iq"], c=v)
        a.set_title(lab); a.set_xlabel("Id [A]"); a.set_ylabel("Iq [A]"); a.grid(True)
        fig.colorbar(sc, ax=a, label=lab)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
