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

Pipeline steps
--------------
The same methods and results as History/preprocess_v3.py:

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
                           FEA grid, and apparent inductances, as in the original

build_flux_map() runs all seven and adds torque.

Changes from the original script
--------------------------------
  * The code is split into functions, and the settings (file, turns ratio,
    stack factor, pole pairs) are arguments rather than constants at the top.
    That lets the same code run on both motors, be tested with pytest and be
    documented with pdoc. Cobalt: turns_ratio=33/2, stack_factor=0.9,
    pole_pairs=5. Zenodo: 1, 1, 8.
  * The Park transform of the inductance matrix is done for all rows at once
    (vectorised). The original row-by-row iterrows loop is kept, selectable
    with inductance_method="loop", so evaluate.py can time the two against
    each other. Both give the same numbers.
  * Steps per electrical cycle use the MEAN rotor-angle step instead of the
    median. Cobalt still gets 16 steps. Zenodo's 60 small steps carry slot
    ripple, which can push the median far enough to round to 59.
  * The incremental-inductance table is assembled without DataFrame.stack(),
    and merged with how="left". The results are the same, but the J = 0 rows
    are kept on every pandas version.
  * Torque now uses the flux formula (derive.torque_from_flux). The original
    matrix-inductance formula is kept as the column Torque_matrix_Nm for
    comparison only (evaluate.torque_formula_comparison).
"""

from __future__ import annotations

__docformat__ = "numpy"   # docstring style, read by pdoc

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
    """Park-transform a three-phase quantity (abc -> dq0).

    Amplitude-invariant (2/3) convention: a balanced set of peak amplitude X
    gives |dq| = X. Used for currents, flux linkages and voltages.

    Parameters
    ----------
    a : float or array_like
        Phase A value.
    b : float or array_like
        Phase B value.
    c : float or array_like
        Phase C value.
    theta : float or array_like
        Rotor electrical angle [rad].

    Returns
    -------
    d : numpy.ndarray or float
        d-axis component.
    q : numpy.ndarray or float
        q-axis component.
    z : numpy.ndarray or float
        Zero-sequence component, (a + b + c) / 3.
    """
    d = (2/3) * (a*np.cos(theta) + b*np.cos(theta - 2*np.pi/3) + c*np.cos(theta + 2*np.pi/3))
    q = -(2/3) * (a*np.sin(theta) + b*np.sin(theta - 2*np.pi/3) + c*np.sin(theta + 2*np.pi/3))
    z = (1/3) * (a + b + c)
    return d, q, z


def park_matrix(theta):
    """Build the 3x3 Park matrix T(theta), so that [d, q, 0] = T @ [a, b, c].

    Parameters
    ----------
    theta : float
        Rotor electrical angle [rad].

    Returns
    -------
    T : numpy.ndarray
        Amplitude-invariant Park matrix, shape (3, 3).
    """
    return (2/3) * np.array([
        [np.cos(theta), np.cos(theta - 2*np.pi/3), np.cos(theta + 2*np.pi/3)],
        [-np.sin(theta), -np.sin(theta - 2*np.pi/3), -np.sin(theta + 2*np.pi/3)],
        [0.5, 0.5, 0.5],
    ])


def dq_inductances_matrix(row, theta):
    """Park-transform one row's 3x3 phase inductance matrix.

    Computes L_dq0 = T @ L_abc @ inv(T) for ONE row. This is the original
    script's per-row version, used by dq_per_row(inductance_method="loop")
    and kept so the vectorised version can be timed against it.

    Parameters
    ----------
    row : pandas.Series
        One row with the working columns Laa, Lab, ... Lcc [H].
    theta : float
        Rotor electrical angle [rad].

    Returns
    -------
    Ld : float
        d-axis inductance [H].
    Lq : float
        q-axis inductance [H].
    """
    T = park_matrix(theta)
    L = np.array([
        [row["Laa"], row["Lab"], row["Lac"]],
        [row["Lba"], row["Lbb"], row["Lbc"]],
        [row["Lca"], row["Lcb"], row["Lcc"]],
    ])
    D = T @ L @ np.linalg.inv(T)
    return D[0, 0], D[1, 1]


def park_matrix_stack(theta):
    """Build the Park matrices for many angles at once.

    Parameters
    ----------
    theta : array_like
        Rotor electrical angles [rad], n values.

    Returns
    -------
    T : numpy.ndarray
        Shape (n, 3, 3); T[k] equals park_matrix(theta[k]).
    """
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

    NaN in the inductance-matrix columns only prints a warning: those
    columns are optional, and their dq values will be NaN.

    Parameters
    ----------
    path : str or pathlib.Path
        Winding-table CSV in the Ansys Maxwell export format, with the
        columns in REQUIRED_COLUMNS.

    Returns
    -------
    raw : pandas.DataFrame
        The file as read, units unchanged.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a required column is missing; a current, flux or setting column
        contains NaN; the row count is not a whole number of operating-point
        blocks (NStepsPerElectricalCycle + 1 rows each); there are no
        no-load (J = 0) rows, which the rotor angle is read from; or the
        phase currents do not sum to zero (an unbalanced or mislabelled set).
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
    """Add the working columns, converting unit prefixes only.

    Inductances uH/nH -> H (Laa ... Lcc) and currents mA -> A (ia, ib, ic);
    flux is already in Wb (fa, fb, fc). The same conversions as the original
    script.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw winding-table rows.

    Returns
    -------
    data : pandas.DataFrame
        The same DataFrame with the working columns added.
    """
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
    """Add the rotor electrical angle "theta" [rad] to every row.

    The original script's logic. With the default rotor_mode "from_flux" the
    angle is read from the no-load flux: at no load the flux points along
    the d-axis, so the rotor angle at each step IS the angle of the no-load
    flux. It is read from the J = 0 blocks and averaged across them (a
    circular mean). Only the first n_keep rows of each block are kept, a
    whole number of electrical cycles, so the averaging in average_cycles()
    is not biased. Cobalt stores 1.5 cycles in 25 rows and keeps 16; Zenodo
    stores 1 cycle in 61 rows and keeps 60.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of load_winding_table().
    rotor_mode : {"from_flux", "fixed", "sweep"}, optional
        How the angle is found. "from_flux" (default) as above. "fixed" uses
        one constant angle, rotor_ref_deg, optionally auto-aligned so the
        no-load psi_q is zero. "sweep" is the legacy method: the angle is
        rebuilt from the step index, which needs init_period_to_remove,
        num_periods and elec_deg_per_cycle to be right.
    rotor_ref_deg : float, optional
        Reference angle [deg] for "fixed" and "sweep". Default 0.
    auto_align : bool, optional
        "fixed" only: shift the angle so the mean no-load psi_q is zero.
        Default True.
    init_period_to_remove : float, optional
        "sweep" only: electrical periods dropped from the start of each
        block. Default 0.5.
    num_periods : float, optional
        "sweep" only: electrical periods stored in each block. Default 1.5.
    elec_deg_per_cycle : float, optional
        "sweep" only: electrical degrees covered by one block's steps, so the
        step is elec_deg_per_cycle / NStepsPerElectricalCycle. Default 360.
    verbose : bool, optional
        Print the detected sweep. Default True.

    Returns
    -------
    data : pandas.DataFrame
        The kept rows, with the working columns and "theta" added.

    Raises
    ------
    ValueError
        If rotor_mode is unknown, or "from_flux" does not find whole no-load
        blocks.
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

    Adds id_raw, iq_raw, psi_d_raw and psi_q_raw (model turns, before
    scaling), and Ldm_raw and Lqm_raw, the dq transform of the Maxwell
    inductance matrix.

    Parameters
    ----------
    data : pandas.DataFrame
        Output of rotor_angle().
    inductance_method : {"vectorised", "loop"}, optional
        "vectorised" (default) computes T @ L @ inv(T) for every row at once.
        "loop" is the original script's row-by-row iterrows loop, slow and
        kept for timing. Both compute exactly the same matrix product.

    Returns
    -------
    data : pandas.DataFrame
        The same DataFrame with the dq columns added.

    Raises
    ------
    ValueError
        If inductance_method is not "vectorised" or "loop".
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
    """Average each operating point over whole electrical cycles.

    Gives one row per operating point (J, TimingAdjustment), holding the
    mean of every kept row. Because rotor_angle() kept a whole number of
    cycles, this mean is the cycle average: the integral over a period
    divided by the period (rectangle rule).

    Parameters
    ----------
    data : pandas.DataFrame
        Output of dq_per_row().

    Returns
    -------
    m : pandas.DataFrame
        One row per operating point: J, TimingAdjustment and the means of
        id_raw, iq_raw, psi_d_raw, psi_q_raw, Ldm_raw and Lqm_raw.
    """
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
    """Scale the 2D model's values to the real machine.

    Flux scales with N*s, current with 1/N and inductance with N^2*s, where
    N is the turns ratio and s the stack factor.

    Parameters
    ----------
    m : pandas.DataFrame
        Output of average_cycles().
    turns_ratio : float
        Real turns / model turns (Cobalt 33/2).
    stack_factor : float
        2D-to-3D correction for field quantities (Cobalt 0.9).

    Returns
    -------
    m : pandas.DataFrame
        The same DataFrame with Id, Iq [A], Psi_d, Psi_q, Psi_mag [Wb] and
        Ld_matrix, Lq_matrix [H] added.
    """
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
    """Find the magnet flux psi_pm and the cross-saturated psi_pm_eff(Iq).

    psi_pm is the no-load psi_d. psi_pm_eff(Iq) is psi_d along the Id = 0
    line: q-axis current saturates iron shared with the magnet flux path, so
    the flux the magnets effectively link drops with |Iq|. It is built from
    the near-zero-Id operating points and interpolated to every row's |Iq|
    (an even function of Iq). The same definition as the original script.

    Parameters
    ----------
    m : pandas.DataFrame
        Output of scale().
    verbose : bool, optional
        Print psi_pm and the cross-saturation. Default True.

    Returns
    -------
    m : pandas.DataFrame
        The same DataFrame with Psi_pm_eff [Wb] added.
    psi_pm : float
        No-load magnet flux linkage [Wb].

    Raises
    ------
    ValueError
        If there are fewer than three operating points near Id = 0.
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
    """Compute incremental and apparent inductances at the FEA points.

    Incremental: finite differences (np.gradient, 3-point central) on the
    polar J x TimingAdjustment grid, then the chain rule (a 2x2 Jacobian
    solve) to get d/dId and d/dIq. They are NaN at J = 0, where the polar
    grid collapses to a point.

    Apparent: Ld_app = (psi_d - psi_pm_eff) / Id and Lq_app = psi_q / Iq.
    The 0/0 points are filled with their limits, the incremental values.

    Parameters
    ----------
    m : pandas.DataFrame
        Output of pm_flux().

    Returns
    -------
    m : pandas.DataFrame
        A new DataFrame with Ld_inc, Ldq_inc, Lqd_inc, Lq_inc, Ld_app and
        Lq_app [H] added, plus Saliency = Lq_matrix / Ld_matrix (the original
        script's exported saliency).
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
    """Add the FEA's own torque, matched on (J, TimingAdjustment).

    The Zenodo FEA computed torque directly. create_zenodo_input_data.py
    saves it, averaged per operating point, in
    zenodo_input_data_operating_points.csv. It is the independent check for
    derive.torque_from_flux(). A warning is printed if some points have no
    match.

    Parameters
    ----------
    m : pandas.DataFrame
        Flux map.
    operating_points_csv : str or pathlib.Path
        The converter's operating-point summary, with column Torque_FEA_Nm.

    Returns
    -------
    m : pandas.DataFrame
        A new DataFrame with Torque_FEA_Nm [Nm] added.
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
    """Turn one winding table into a flux map (steps 1-7, then torque).

    Parameters
    ----------
    path : str or pathlib.Path
        Winding-table CSV in the Maxwell export format.
    turns_ratio : float
        Real turns / model turns (Cobalt 33/2, Zenodo 1).
    stack_factor : float
        2D-to-3D correction for field quantities (Cobalt 0.9, Zenodo 1).
    pole_pairs : int
        Number of pole pairs p (Cobalt 5, Zenodo 8).
    rotor_mode : {"from_flux", "fixed", "sweep"}, optional
        How the rotor angle is found; see rotor_angle(). Default "from_flux".
    inductance_method : {"vectorised", "loop"}, optional
        How the inductance matrix is Park-transformed; see dq_per_row().
        Default "vectorised".
    fea_torque_csv : str or pathlib.Path, optional
        Operating-point file with the FEA's own torque (Zenodo only). Adds
        the column Torque_FEA_Nm. Default None.
    verbose : bool, optional
        Print the rotor sweep, psi_pm and the inductance comparison. Default
        True.
    **rotor_kwargs
        Further options passed to rotor_angle().

    Returns
    -------
    flux_map : pandas.DataFrame
        One row per FEA operating point. J [A_per_m2] and TimingAdjustment
        [deg] are the FEA sweep settings; Id and Iq [A] the real-machine dq
        current (peak); Psi_d, Psi_q, Psi_mag, Psi_pm_eff and Psi_pm [Wb] the
        flux linkages; Ld_matrix and Lq_matrix [H] Maxwell's matrix,
        dq-transformed; Ld_inc, Ldq_inc, Lqd_inc and Lq_inc [H] finite
        differences on the FEA grid; Ld_app and Lq_app [H] the apparent
        inductances; Saliency = Lq_matrix / Ld_matrix; Torque_Nm the flux
        formula; Torque_matrix_Nm the original matrix formula (for
        comparison); Imag [A] and beta [deg] the current magnitude and angle;
        and Torque_FEA_Nm if fea_torque_csv is given. The cycle-averaged
        model-turn values (id_raw ... Lqm_raw) are kept too.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the table fails a check in load_winding_table(), rotor_angle(),
        dq_per_row() or pm_flux().

    Examples
    --------
    >>> flux_map = build_flux_map("Input_Data/Cobalt/cobalt_input_data.csv",
    ...                           turns_ratio=33 / 2, stack_factor=0.9,
    ...                           pole_pairs=5, verbose=False)
    >>> len(unique_operating_points(flux_map))
    71
    >>> print(f"psi_pm = {flux_map['Psi_pm'].iloc[0] * 1e3:.1f} mWb")
    psi_pm = 48.2 mWb
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
    """Print the original script's comparison of the inductance definitions.

    Shows the range of the incremental, apparent and matrix Ld and Lq, the
    cross-coupling, and the peak torque by both torque formulas.

    Parameters
    ----------
    m : pandas.DataFrame
        Flux map from build_flux_map().
    """
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
    repeat in both the training and test folds would make cross-validation
    look better than it is, so only the first copy is kept.

    Parameters
    ----------
    m : pandas.DataFrame
        Flux map from build_flux_map().

    Returns
    -------
    pts : pandas.DataFrame
        The unique operating points, re-indexed from 0.
    """
    t = ((m[COL_TIMING] + 180.0) % 360.0) - 180.0               # -180 and +180 -> same
    t = np.where(m[COL_J] == 0, 0.0, t)                         # every J=0 row is one point
    key = pd.DataFrame({"J": m[COL_J].round(9).to_numpy(), "T": np.round(t, 6)})
    return m.loc[~key.duplicated().to_numpy()].reset_index(drop=True)


def base_values(m, pole_pairs):
    """Compute the per-unit bases used to compare errors across motors.

    Parameters
    ----------
    m : pandas.DataFrame
        Flux map from build_flux_map().
    pole_pairs : int
        Number of pole pairs p.

    Returns
    -------
    bases : dict of float
        "I_base" = the largest |I| [A], "psi_base" = psi_pm [Wb] and
        "T_base" = 1.5 * p * psi_pm * I_base [Nm].
    """
    I_base = float(np.hypot(m["Id"], m["Iq"]).max())
    psi_base = float(m["Psi_pm"].iloc[0])
    return {"I_base": I_base, "psi_base": psi_base,
            "T_base": 1.5 * pole_pairs * psi_base * I_base}


def plot_flux_map(m, title="", path=None, show=False):
    """Plot six quantities of the flux map as filled colour maps over Id-Iq.

    The panels are the original script's six: Maxwell's matrix Ld and Lq,
    saliency, |psi|, torque (from the flux formula) and psi_pm_eff. The
    colour between FEA points is a straight-line blend of the neighbouring
    points (a triangulation, no fitted model), so the figure shows only the
    FEA data; the points are drawn on top. Signed quantities (Zenodo's
    torque, which runs in all four quadrants) use a red-blue scale centred
    on zero. Everything else uses a light single-hue scale so the dark end
    never hides the points.

    Parameters
    ----------
    m : pandas.DataFrame
        Flux map from build_flux_map().
    title : str, optional
        Figure title. Default "" (no title).
    path : str or pathlib.Path, optional
        Save the figure here as a PNG. Default None (not saved).
    show : bool, optional
        Also show the figure in a window. Default False.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")                   # draw off-screen (no window)
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    ink, quiet = "#1f2328", "#59636e"           # text colour, axis/point colour
    light_blues = LinearSegmentedColormap.from_list(
        "light_blues", plt.cm.Blues(np.linspace(0.04, 0.62, 256)))
    red_blue = LinearSegmentedColormap.from_list(
        "light_red_blue", plt.cm.RdBu_r(np.linspace(0.12, 0.88, 256)))

    # One value per operating point (repeated no-load points are averaged)
    pts = m.groupby(["Id", "Iq"], as_index=False).mean(numeric_only=True)
    tri = mtri.Triangulation(pts["Id"], pts["Iq"])
    dense = len(pts) > 300                      # Zenodo: full-size dots would hide the colours

    panels = [(pts["Ld_matrix"] * 1e6, "Ld matrix [µH]"),
              (pts["Lq_matrix"] * 1e6, "Lq matrix [µH]"),
              (pts["Saliency"], "Saliency Lq/Ld [–]"),
              (pts["Psi_mag"] * 1e3, "|ψ| [mWb]"),
              (pts["Torque_Nm"], "Torque [Nm]"),
              (pts["Psi_pm_eff"] * 1e3, "ψpm,eff [mWb]")]

    style = {"font.size": 10, "axes.edgecolor": quiet, "axes.labelcolor": ink,
             "xtick.color": quiet, "ytick.color": quiet,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, ax = plt.subplots(2, 3, figsize=(15, 9.4))
        for a, (v, lab) in zip(ax.flat, panels):
            # Colour range from the 1st to 99th percentile, so a few extreme
            # points don't wash out the rest (they show in the end arrows)
            lo, hi = np.percentile(v, [1, 99])
            signed = lo < 0 < hi and min(-lo, hi) > 0.1 * max(-lo, hi)
            if signed:                          # centre the scale on zero
                lo, hi = -max(-lo, hi), max(-lo, hi)
            levels = MaxNLocator(nbins=12).tick_values(lo, hi)   # round-number bands
            cf = a.tricontourf(tri, v, levels=levels, extend="both",
                               cmap=red_blue if signed else light_blues)
            if dense:
                a.scatter(pts["Id"], pts["Iq"], s=0.6, color=quiet, alpha=0.5, lw=0)
            else:
                a.scatter(pts["Id"], pts["Iq"], s=7, facecolor="white", edgecolor=quiet, lw=0.5)
            a.set(xlabel="Id [A]", ylabel="Iq [A]", aspect="equal")
            a.set_title(lab, loc="left", color=ink, fontsize=11)
            fig.colorbar(cf, ax=a, shrink=0.8)
        if title:
            fig.suptitle(title, x=0.01, ha="left", color=ink, fontsize=13)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
