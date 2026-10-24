"""
create_zenodo_input_data.py - turn the open Zenodo IPM maps into a winding table.

Where this fits in the project
------------------------------
The project's script is developed on the Cobalt FEA and validated, unchanged, on
a second, independent motor. That motor's data comes from Zenodo as dq maps
(Psi_d.csv, Psi_q.csv, Torque.csv), not in the Maxwell winding-table format the
script reads. This file rebuilds the Maxwell columns from the dq maps and writes

    Input_Data/Zenodo_IPM/zenodo_input_data.csv                   same columns as cobalt_input_data.csv
    Input_Data/Zenodo_IPM/zenodo_input_data_operating_points.csv  FEA torque per operating point
    Input_Data/Zenodo_IPM/zenodo_input_data_provenance.json       how every column was made

so preprocess.py can read both motors the same way. run_all.py calls it
automatically if zenodo_input_data.csv is missing.

Source dataset (CC BY 4.0):
    Mai, Liu, Wagner, Hofmann, "Current Harmonics Minimization of PMSM Based on
    Iterative Learning Control and Neural Networks: Motor Data", Zenodo (2023),
    https://doi.org/10.5281/zenodo.7866449
Cite the associated paper when using the data (the dataset asks for this):
    Machines 2023, 11(8), 784, https://doi.org/10.3390/machines11080784

Source layout: three CSVs (Psi_d, Psi_q, Torque). Each has a time column plus
2331 operating-point columns named "... - Gamma='<deg>' im='<A>A'". Each column
is one electrical rotation (61 samples, both endpoints included). Im is the
peak phase current, 0-620 A in 10 A steps. Gamma is the current angle,
-180..180 deg in 10 deg steps: Gamma = 0 is pure +q and Gamma = +90 is pure -d,
so Id = -Im*sin(Gamma) and Iq = Im*cos(Gamma). This is the same convention as
TimingAdjustment in the Maxwell winding table. The dq quantities are
amplitude-invariant (peak): the FEA torque matches 1.5*p*(psi_d*iq - psi_q*id)
with p = 8 to ~0.05 Nm RMS.

How each output column is produced (also written to a provenance JSON):
    J [A_per_m2]                  PROXY. Im / conductor_area. The conductor area
                                  is not in the dataset, so the default area
                                  of 1 m^2 makes J numerically equal to Im [A].
                                  J is exactly 0 at no load, which the analysis
                                  script uses to find no-load blocks.
    NStepsPerElectricalCycle []   Samples per block minus one (60).
    RPM [rpm]                     Derived from the time span of one electrical
                                  rotation and the pole pairs.
    TimingAdjustment [deg]        = Gamma (same convention).
    Time [s]                      From the source time column (resets each block).
    L(Phase_x,Phase_y) [uH]/[nH]  DERIVED (the dataset has no inductance matrix).
                                  Incremental inductance dpsi_abc/di_abc at each
                                  rotor position: finite differences of psi_d,
                                  psi_q across the (Im, Gamma) grid give the dq
                                  matrix [[Ldd, Ldq], [Lqd, Lqq]]. The cross terms
                                  are symmetrised (reciprocity), then mapped to
                                  abc with the inverse Park transform.
                                  Zero-sequence inductance is set to 0 (not in
                                  the data; it does not affect Ld or Lq).
                                  These are NOT Maxwell's frozen-permeability
                                  values, so for this motor the "matrix" and
                                  "incremental" inductances in the analysis
                                  script agree by construction. Use
                                  --inductance none to leave the columns empty.
    FluxLinkage(Phase_x) [Wb]     RECONSTRUCTED by inverse Park transform of
                                  psi_d, psi_q (zero-sequence flux assumed 0).
    InducedVoltage(Phase_x) [uv]  DERIVED as d(psi_x)/dt by periodic central
                                  differences (Maxwell reports +dpsi/dt).
    InputCurrent(Phase_x) [mA]    RECONSTRUCTED by inverse Park transform of
                                  the constant (Id, Iq) of each operating point.

When analysing the output with the winding-table script, use
turns_ratio = 1, stack_factor = 1, pole_pairs = 8. The data is already for the
full machine. Your own motor's 33/2 and 0.9 corrections do not apply.

Usage (defaults assume the repo layout Scripts/, Input_Data/Zenodo_IPM/):
    python Scripts/create_zenodo_input_data.py
    python Scripts/create_zenodo_input_data.py --psi-d path/Psi_d.csv --psi-q path/Psi_q.csv \
        --torque path/Torque.csv --out path/zenodo_input_data.csv
or from Python:  create_zenodo_input_data(psi_d_path, psi_q_path, torque_path, out_path)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLE_PAIRS_DEFAULT = 8  # 24 slots / 16 poles (dataset description)

# Column order of the Maxwell winding-table export (copied from Winding Table 1.csv)
WINDING_TABLE_COLUMNS = [
    "J [A_per_m2]",
    "NStepsPerElectricalCycle []",
    "RPM [rpm]",
    "TimingAdjustment [deg]",
    "Time [s]",
    "L(Phase_A,Phase_A) [uH]",
    "L(Phase_B,Phase_A) [nH]",
    "L(Phase_C,Phase_A) [nH]",
    "L(Phase_A,Phase_B) [nH]",
    "L(Phase_B,Phase_B) [uH]",
    "L(Phase_C,Phase_B) [nH]",
    "L(Phase_A,Phase_C) [nH]",
    "L(Phase_B,Phase_C) [nH]",
    "L(Phase_C,Phase_C) [uH]",
    "FluxLinkage(Phase_A) [Wb]",
    "FluxLinkage(Phase_B) [Wb]",
    "FluxLinkage(Phase_C) [Wb]",
    "InducedVoltage(Phase_A) [uv]",
    "InducedVoltage(Phase_B) [uv]",
    "InducedVoltage(Phase_C) [uv]",
    "InputCurrent(Phase_A) [mA]",
    "InputCurrent(Phase_B) [mA]",
    "InputCurrent(Phase_C) [mA]",
]

_HEADER_RE = re.compile(r"Gamma='(-?\d+(?:\.\d+)?)'\s+im='(\d+(?:\.\d+)?)A'")
_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


# ---------------------------------------------------------------------------
# Transforms (amplitude-invariant, identical convention to park_vec() in the
# winding-table analysis script)
# ---------------------------------------------------------------------------
def park(a, b, c, theta):
    """abc -> dq (amplitude-invariant). Same as park_vec() in the analysis script."""
    d = (2 / 3) * (a * np.cos(theta) + b * np.cos(theta - 2 * np.pi / 3)
                   + c * np.cos(theta + 2 * np.pi / 3))
    q = -(2 / 3) * (a * np.sin(theta) + b * np.sin(theta - 2 * np.pi / 3)
                    + c * np.sin(theta + 2 * np.pi / 3))
    return d, q


def inverse_park(d, q, theta):
    """dq -> abc (amplitude-invariant, zero-sequence = 0). Exact inverse of park()."""
    a = d * np.cos(theta) - q * np.sin(theta)
    b = d * np.cos(theta - 2 * np.pi / 3) - q * np.sin(theta - 2 * np.pi / 3)
    c = d * np.cos(theta + 2 * np.pi / 3) - q * np.sin(theta + 2 * np.pi / 3)
    return a, b, c


def incremental_inductance_dq(im_vals, gamma_deg, psi_d, psi_q):
    """Incremental dq inductances from flux maps on a full polar current grid.

    Parameters
    ----------
    im_vals : (n_im,) current magnitudes, ascending, must start at 0
    gamma_deg : (n_g,) current angles, uniformly spaced, covering a full circle
        WITHOUT the duplicate endpoint (e.g. -180..170 in 10 deg steps)
    psi_d, psi_q : (n_t, n_im, n_g) flux linkage at each rotor sample

    Returns
    -------
    Ldd, Ldq, Lqd, Lqq : (n_t, n_im, n_g) arrays, H
        Ldd = dpsi_d/dId, Ldq = dpsi_d/dIq, Lqd = dpsi_q/dId, Lqq = dpsi_q/dIq

    Method
    ------
    Derivatives along the grid axes (np.gradient in Im, 2nd order; periodic
    4th-order central differences in Gamma) are converted to d/dId, d/dIq with the exact Jacobian
    of Id = -Im sin(g), Iq = Im cos(g):
        d/dId = -sin(g) d/dIm - cos(g)/Im d/dg
        d/dIq =  cos(g) d/dIm - sin(g)/Im d/dg
    At Im = 0 the polar Jacobian is singular, so Cartesian central differences
    across the first ring are used (+-Id at g = -/+90 deg, +-Iq at g = 0/180).
    """
    if im_vals[0] != 0:
        raise ValueError("Inductance derivation needs an Im = 0 point")
    dg_deg = np.diff(gamma_deg)
    if not np.allclose(dg_deg, dg_deg[0]) or not np.isclose(dg_deg[0] * len(gamma_deg), 360):
        raise ValueError("Gamma must be uniformly spaced over a full circle (no duplicate endpoint)")
    g = np.deg2rad(gamma_deg)[None, None, :]
    dg = np.deg2rad(dg_deg[0])
    s, c = np.sin(g), np.cos(g)
    I = im_vals[None, :, None].astype(float)

    def to_dq(x):
        x_im = np.gradient(x, im_vals, axis=1, edge_order=2)
        # 4th-order periodic central difference in Gamma. The 2nd-order stencil
        # underestimates the angular derivative by sin(h)/h (~0.5 % at 10 deg).
        r = lambda k: np.roll(x, -k, axis=2)
        x_g = (-r(2) + 8 * r(1) - 8 * r(-1) + r(-2)) / (12 * dg)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -s * x_im - c / I * x_g, c * x_im - s / I * x_g

    Ldd, Ldq = to_dq(psi_d)
    Lqd, Lqq = to_dq(psi_q)

    # Origin: Cartesian central differences on the first ring
    r = im_vals[1]
    k = lambda deg: int(np.argmin(np.abs((gamma_deg - deg + 180) % 360 - 180)))
    pd_, md_, pq_, mq_ = k(-90), k(90), k(0), k(180)     # +Id, -Id, +Iq, -Iq
    for L_id, L_iq, psi in ((Ldd, Ldq, psi_d), (Lqd, Lqq, psi_q)):
        L_id[:, 0, :] = ((psi[:, 1, pd_] - psi[:, 1, md_]) / (2 * r))[:, None]
        L_iq[:, 0, :] = ((psi[:, 1, pq_] - psi[:, 1, mq_]) / (2 * r))[:, None]
    return Ldd, Ldq, Lqd, Lqq


def dq_to_abc_inductance(Ldd, Lqq, Lc, theta):
    """3x3 phase inductance matrix from a symmetric dq matrix [[Ldd, Lc], [Lc, Lqq]]
    with zero-sequence inductance 0 (amplitude-invariant transform):
        L_xy = 2/3 * (Ldd cos(tx) cos(ty) + Lqq sin(tx) sin(ty) - Lc sin(tx + ty))
    with tx = theta, theta - 2pi/3, theta + 2pi/3 for phases A, B, C.
    Park-transforming this matrix (T @ L @ inv(T)) returns Ldd, Lqq and Lc exactly.
    Returns a dict keyed by phase pairs, e.g. ('A', 'B').
    """
    tx = {"A": theta, "B": theta - 2 * np.pi / 3, "C": theta + 2 * np.pi / 3}
    out = {}
    for x in "ABC":
        for y in "ABC":
            out[(x, y)] = (2 / 3) * (Ldd * np.cos(tx[x]) * np.cos(tx[y])
                                     + Lqq * np.sin(tx[x]) * np.sin(tx[y])
                                     - Lc * np.sin(tx[x] + tx[y]))
    return out


# ---------------------------------------------------------------------------
# Reading and validating the Zenodo files
# ---------------------------------------------------------------------------
def read_zenodo_map(path: Path):
    """Read one Zenodo map CSV.

    Returns
    -------
    time : (n_t,) array in the file's time unit
    time_unit : str, unit parsed from the time column header (e.g. 'ms')
    gamma_deg, im_a : (n_op,) arrays parsed from the column headers
    values : (n_t, n_op) array
    """
    df = pd.read_csv(path)
    tcol = df.columns[0]
    if not tcol.lower().startswith("time"):
        raise ValueError(f"{path.name}: first column should be time, got '{tcol}'")
    m_unit = re.search(r"\[(\w+)\]", tcol)
    time_unit = m_unit.group(1) if m_unit else "s"

    meta = []
    for col in df.columns[1:]:
        m = _HEADER_RE.search(col)
        if m is None:
            raise ValueError(f"{path.name}: cannot parse Gamma/Im from column '{col}'")
        meta.append((float(m.group(1)), float(m.group(2))))
    meta = np.array(meta)

    values = df.iloc[:, 1:].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"{path.name}: contains NaN values")
    return df[tcol].to_numpy(dtype=float), time_unit, meta[:, 0], meta[:, 1], values


def load_zenodo(psi_d_path: Path, psi_q_path: Path, torque_path: Path | None):
    """Load and cross-check the Zenodo CSVs. Returns a dict of arrays."""
    t, unit, g, im, psi_d = read_zenodo_map(psi_d_path)
    t2, _, g2, im2, psi_q = read_zenodo_map(psi_q_path)
    if not (np.allclose(t, t2) and np.array_equal(g, g2) and np.array_equal(im, im2)):
        raise ValueError("Psi_d and Psi_q files do not share the same time base / operating points")

    torque = None
    if torque_path is not None:
        t3, _, g3, im3, torque = read_zenodo_map(torque_path)
        if not (np.allclose(t, t3) and np.array_equal(g, g3) and np.array_equal(im, im3)):
            raise ValueError("Torque file does not match the flux files")

    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-6):
        raise ValueError("Time samples are not uniformly spaced")
    if not np.any(im == 0):
        raise ValueError("No Im = 0 (no-load) operating points: the analysis script needs them")
    return dict(time=t, time_unit=unit, gamma=g, im=im, psi_d=psi_d, psi_q=psi_q, torque=torque)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def _inductance_columns(im, g, psi_d, psi_q, theta):
    """Fill the 9 L(...) columns with the derived incremental inductance matrix.

    im, g : (n_op,) operating points sorted Im outer, Gamma inner (full grid)
    psi_d, psi_q : (n_t, n_op);  theta : (n_t, 1)
    Returns (dict column -> (n_t, n_op) array in the column's unit, stats dict)
    """
    im_vals, g_vals = np.unique(im), np.unique(g)
    if len(im_vals) * len(g_vals) != len(im):
        raise ValueError("Operating points do not form a complete Im x Gamma grid")
    n_t = psi_d.shape[0]
    grid = lambda x: x.reshape(n_t, len(im_vals), len(g_vals))
    # drop the duplicate +180 deg column for the periodic derivative, map it back afterwards
    dup = np.isclose(g_vals[-1] - g_vals[0], 360)
    g_red = g_vals[:-1] if dup else g_vals
    n_red = len(g_red)
    Ldd, Ldq, Lqd, Lqq = incremental_inductance_dq(
        im_vals, g_red, grid(psi_d)[:, :, :n_red], grid(psi_q)[:, :, :n_red])
    back = np.array([int(np.argmin(np.abs((g_red - gg + 180) % 360 - 180))) for gg in g_vals])
    full = lambda L: L[:, :, back].reshape(n_t, -1)
    Ldd, Ldq, Lqd, Lqq = full(Ldd), full(Ldq), full(Lqd), full(Lqq)

    Lc = 0.5 * (Ldq + Lqd)                       # reciprocity: dpsi_d/diq = dpsi_q/did
    asym = np.abs(Ldq - Lqd)
    stats = {
        "Ld_incremental_uH_range": [float(Ldd.min() * 1e6), float(Ldd.max() * 1e6)],
        "Lq_incremental_uH_range": [float(Lqq.min() * 1e6), float(Lqq.max() * 1e6)],
        "Ldq_cross_uH_range": [float(Lc.min() * 1e6), float(Lc.max() * 1e6)],
        "reciprocity_|Ldq-Lqd|_median_uH": float(np.median(asym) * 1e6),
        "reciprocity_|Ldq-Lqd|_p95_uH": float(np.percentile(asym, 95) * 1e6),
    }

    Labc = dq_to_abc_inductance(Ldd, Lqq, Lc, theta)
    cols = {}
    for col in WINDING_TABLE_COLUMNS:
        m = re.match(r"L\(Phase_([ABC]),Phase_([ABC])\) \[(uH|nH)\]", col)
        if m:
            scale = 1e6 if m.group(3) == "uH" else 1e9
            cols[col] = Labc[(m.group(1), m.group(2))] * scale
    return cols, stats, (Ldd, Lqq, Lc)


def build_winding_table(z: dict, pole_pairs: int, conductor_area_m2: float,
                        time_unit: str | None = None, inductance: str = "incremental"):
    """Build a DataFrame with the Maxwell winding-table columns.

    Each operating point becomes one block of n_t rows (one electrical rotation),
    ordered J (current) outer and TimingAdjustment (angle) inner, as in the
    Maxwell export. inductance = 'incremental' fills the L(...) columns with the
    derived incremental matrix; 'none' leaves them empty.
    """
    unit = time_unit or z["time_unit"]
    if unit not in _TIME_UNITS:
        raise ValueError(f"Unknown time unit '{unit}'")
    t_s = (z["time"] - z["time"][0]) * _TIME_UNITS[unit]   # seconds from block start
    n_t = len(t_s)
    period_s = t_s[-1]                                      # one electrical rotation
    theta = (2 * np.pi * t_s / period_s)[:, None]           # rotor electrical angle, (n_t, 1)
    rpm = 60.0 / (period_s * pole_pairs)

    # Sort operating points: current magnitude outer, angle inner
    order = np.lexsort((z["gamma"], z["im"]))
    g, im = z["gamma"][order], z["im"][order]
    psi_d, psi_q = z["psi_d"][:, order], z["psi_q"][:, order]

    gr = np.deg2rad(g)
    i_d, i_q = -im * np.sin(gr), im * np.cos(gr)            # peak, amplitude-invariant

    # Phase currents and flux linkages, shape (n_t, n_op)
    ia, ib, ic = inverse_park(i_d[None, :], i_q[None, :], theta)
    fa, fb, fc = inverse_park(psi_d, psi_q, theta)

    # Induced voltage e = d(psi)/dt, periodic central differences over the
    # n_t-1 unique samples (first and last sample are the same rotor angle)
    dt = t_s[1] - t_s[0]

    def ddt(f):
        u = f[:-1]
        e = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dt)
        return np.vstack([e, e[:1]])

    ea, eb, ec = ddt(fa), ddt(fb), ddt(fc)

    n_op = len(g)
    flat = lambda x: np.asarray(x).T.reshape(-1)            # block-major (op, time)
    rep = lambda v: np.repeat(v, n_t)

    out = pd.DataFrame({
        "J [A_per_m2]": rep(im / conductor_area_m2),
        "NStepsPerElectricalCycle []": n_t - 1,
        "RPM [rpm]": rpm,
        "TimingAdjustment [deg]": rep(g),
        "Time [s]": np.tile(t_s, n_op),
    })
    ind_stats, ind_dq = {}, None
    if inductance == "incremental":
        ind_cols, ind_stats, ind_dq = _inductance_columns(im, g, psi_d, psi_q, theta)
        for col, val in ind_cols.items():
            out[col] = flat(val)
    elif inductance == "none":
        for col in WINDING_TABLE_COLUMNS:
            if col.startswith("L("):
                out[col] = np.nan
    else:
        raise ValueError(f"Unknown inductance option '{inductance}'")
    out["FluxLinkage(Phase_A) [Wb]"] = flat(fa)
    out["FluxLinkage(Phase_B) [Wb]"] = flat(fb)
    out["FluxLinkage(Phase_C) [Wb]"] = flat(fc)
    out["InducedVoltage(Phase_A) [uv]"] = flat(ea) * 1e6
    out["InducedVoltage(Phase_B) [uv]"] = flat(eb) * 1e6
    out["InducedVoltage(Phase_C) [uv]"] = flat(ec) * 1e6
    out["InputCurrent(Phase_A) [mA]"] = flat(ia) * 1e3
    out["InputCurrent(Phase_B) [mA]"] = flat(ib) * 1e3
    out["InputCurrent(Phase_C) [mA]"] = flat(ic) * 1e3
    out = out[WINDING_TABLE_COLUMNS]

    # Operating-point summary (one row per block), incl. FEA torque if available
    keep = slice(0, n_t - 1)                                 # one full cycle, no duplicate endpoint
    summary = pd.DataFrame({
        "J [A_per_m2]": im / conductor_area_m2,
        "TimingAdjustment [deg]": g,
        "Im_A": im,
        "Gamma_deg": g,
        "Id_A": i_d,
        "Iq_A": i_q,
        "Psi_d_Wb": psi_d[keep].mean(axis=0),
        "Psi_q_Wb": psi_q[keep].mean(axis=0),
    })
    if z["torque"] is not None:
        tq = z["torque"][:, order]
        summary["Torque_FEA_Nm"] = tq[keep].mean(axis=0)
        summary["Torque_ripple_pkpk_Nm"] = np.ptp(tq[keep], axis=0)
    if ind_dq is not None:
        Ldd, Lqq, Lc = ind_dq
        summary["Ld_inc_uH"] = Ldd[keep].mean(axis=0) * 1e6
        summary["Lq_inc_uH"] = Lqq[keep].mean(axis=0) * 1e6
        summary["Ldq_inc_uH"] = Lc[keep].mean(axis=0) * 1e6
    return out, summary, dict(rpm=rpm, period_s=period_s, n_t=n_t, n_op=n_op, time_unit=unit,
                              inductance=inductance, inductance_stats=ind_stats)


def self_check(table: pd.DataFrame, summary: pd.DataFrame, n_t: int, pole_pairs: int):
    """Round-trip check: Park-transform the written abc columns back to dq and
    compare with the source values. Returns a dict of max errors."""
    th = np.tile(2 * np.pi * np.arange(n_t) / (n_t - 1), len(summary))
    d_f, q_f = park(table["FluxLinkage(Phase_A) [Wb]"].to_numpy(),
                    table["FluxLinkage(Phase_B) [Wb]"].to_numpy(),
                    table["FluxLinkage(Phase_C) [Wb]"].to_numpy(), th)
    d_i, q_i = park(table["InputCurrent(Phase_A) [mA]"].to_numpy() * 1e-3,
                    table["InputCurrent(Phase_B) [mA]"].to_numpy() * 1e-3,
                    table["InputCurrent(Phase_C) [mA]"].to_numpy() * 1e-3, th)
    blk = lambda x: x.reshape(-1, n_t)[:, :-1].mean(axis=1)
    res = {
        "max_err_psi_d_Wb": float(np.max(np.abs(blk(d_f) - summary["Psi_d_Wb"]))),
        "max_err_psi_q_Wb": float(np.max(np.abs(blk(q_f) - summary["Psi_q_Wb"]))),
        "max_err_id_A": float(np.max(np.abs(blk(d_i) - summary["Id_A"]))),
        "max_err_iq_A": float(np.max(np.abs(blk(q_i) - summary["Iq_A"]))),
    }
    if "Ld_inc_uH" in summary:
        # Park-transform the written 3x3 matrices (T @ L @ inv(T), as the analysis
        # script does) and compare with the dq inductances they were built from
        n = len(table)
        c0, c1, c2 = np.cos(th), np.cos(th - 2 * np.pi / 3), np.cos(th + 2 * np.pi / 3)
        s0, s1, s2 = np.sin(th), np.sin(th - 2 * np.pi / 3), np.sin(th + 2 * np.pi / 3)
        T = (2 / 3) * np.stack([np.stack([c0, c1, c2], -1),
                                np.stack([-s0, -s1, -s2], -1),
                                np.full((n, 3), 0.5)], 1)
        u, nn = 1e-6, 1e-9
        col = lambda x, y, unit: table[f"L(Phase_{x},Phase_{y}) [{unit}]"].to_numpy()
        L = np.empty((n, 3, 3))
        for i, x in enumerate("ABC"):
            for j, y in enumerate("ABC"):
                L[:, i, j] = col(x, y, "uH") * u if x == y else col(x, y, "nH") * nn
        D = T @ L @ np.linalg.inv(T)
        res["max_err_Ld_from_matrix_uH"] = float(np.max(np.abs(blk(D[:, 0, 0]) * 1e6 - summary["Ld_inc_uH"])))
        res["max_err_Lq_from_matrix_uH"] = float(np.max(np.abs(blk(D[:, 1, 1]) * 1e6 - summary["Lq_inc_uH"])))
    if "Torque_FEA_Nm" in summary:
        t_flux = 1.5 * pole_pairs * (summary["Psi_d_Wb"] * summary["Iq_A"]
                                     - summary["Psi_q_Wb"] * summary["Id_A"])
        err = t_flux - summary["Torque_FEA_Nm"]
        res["torque_flux_vs_FEA_rms_Nm"] = float(np.sqrt(np.mean(err ** 2)))
        res["torque_FEA_max_Nm"] = float(summary["Torque_FEA_Nm"].max())
    return res


def provenance(params, info, checks):
    """Per-column provenance record written next to the output CSV."""
    return {
        "source": "Zenodo record 7866449 (CC BY 4.0), https://doi.org/10.5281/zenodo.7866449",
        "cite": "Machines 2023, 11(8), 784, https://doi.org/10.3390/machines11080784",
        "converter": Path(__file__).name,
        "parameters": {
            "pole_pairs": params["pole_pairs"],
            "conductor_area_m2 (J proxy)": params["conductor_area"],
            "time_unit_used": info["time_unit"],
            "electrical_period_s": info["period_s"],
            "rpm_derived": info["rpm"],
            "samples_per_block": info["n_t"],
            "operating_points": info["n_op"],
        },
        "analysis_settings_for_this_file": {"turns_ratio": 1, "stack_factor": 1,
                                            "pole_pairs": params["pole_pairs"]},
        "columns": {
            "J [A_per_m2]": "PROXY: Im / conductor_area (area not in dataset)",
            "NStepsPerElectricalCycle []": "samples per block - 1",
            "RPM [rpm]": "DERIVED from time span and pole pairs",
            "TimingAdjustment [deg]": "= Gamma (0 = +q, 90 = -d)",
            "Time [s]": "SOURCE time column",
            "L(Phase_x,Phase_y)": (
                "DERIVED: incremental inductance dpsi_abc/di_abc per rotor sample, from finite "
                "differences of SOURCE psi_d, psi_q over the (Im, Gamma) grid; cross terms "
                "symmetrised; zero-sequence = 0. NOT Maxwell frozen-permeability values."
                if info["inductance"] == "incremental" else "NOT AVAILABLE (empty)"),
            "FluxLinkage(Phase_x) [Wb]": "RECONSTRUCTED: inverse Park of SOURCE psi_d, psi_q; zero-sequence = 0",
            "InducedVoltage(Phase_x) [uv]": "DERIVED: periodic central difference of reconstructed phase flux",
            "InputCurrent(Phase_x) [mA]": "RECONSTRUCTED: inverse Park of Id=-Im*sin(Gamma), Iq=Im*cos(Gamma)",
        },
        "inductance_stats": info["inductance_stats"],
        "self_check": checks,
        "notes": [
            "Source readme says one electrical rotation lasts 2.5 s, but the CSV time "
            "header is in ms (0-2.5 ms). The header is used by default (3000 rpm); "
            "use --time-unit s for the readme's value (3 rpm). Only RPM and "
            "InducedVoltage depend on this.",
            "Gamma = -180 and +180 are the same physical operating point; Im = 0 is "
            "repeated for every Gamma. Both duplicates are kept, as in the source.",
        ],
    }


# ---------------------------------------------------------------------------
# MAKE THE FILES (callable from run_all.py) AND COMMAND LINE
# ---------------------------------------------------------------------------
def create_zenodo_input_data(psi_d_path, psi_q_path, torque_path, out_path,
                             pole_pairs=POLE_PAIRS_DEFAULT, conductor_area=1.0,
                             time_unit=None, inductance="incremental", verbose=True):
    """Convert the Zenodo CSVs and write the three output files.

    psi_d_path, psi_q_path : Zenodo Psi_d.csv and Psi_q.csv
    torque_path            : Zenodo Torque.csv, or None to skip the FEA torque
    out_path               : the winding table to write (zenodo_input_data.csv).
                             The operating-point summary and provenance file are
                             written beside it.
    Returns a dict of the self-check results.
    """
    psi_d_path, psi_q_path, out_path = Path(psi_d_path), Path(psi_q_path), Path(out_path)
    missing = [f for f in (psi_d_path, psi_q_path) if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Could not find the Zenodo flux files:\n  " + "\n  ".join(str(f) for f in missing)
            + "\nPut Psi_d.csv, Psi_q.csv and Torque.csv in Input_Data/Zenodo_IPM/ "
              "(or pass the paths, e.g. --psi-d Psi_d.csv --psi-q Psi_q.csv --torque Torque.csv)")
    torque_path = Path(torque_path) if torque_path is not None else None
    if torque_path is not None and not torque_path.exists():
        print(f"Note: no torque file at {torque_path}; skipping the FEA torque summary/check.")
        torque_path = None

    # Read, convert, check
    z = load_zenodo(psi_d_path, psi_q_path, torque_path)
    table, summary, info = build_winding_table(z, pole_pairs, conductor_area, time_unit, inductance)
    checks = self_check(table, summary, info["n_t"], pole_pairs)

    # Write the winding table, the operating-point summary and the provenance record
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    summary_path = out_path.with_name(out_path.stem + "_operating_points.csv")
    summary.to_csv(summary_path, index=False)
    prov_path = out_path.with_name(out_path.stem + "_provenance.json")
    params = {"pole_pairs": pole_pairs, "conductor_area": conductor_area}
    prov_path.write_text(json.dumps(provenance(params, info, checks), indent=2))

    if verbose:
        print(f"Wrote {out_path}  ({len(table)} rows = {info['n_op']} operating points x {info['n_t']} samples)")
        print(f"Wrote {summary_path}")
        print(f"Wrote {prov_path}")
        print(f"Derived speed: {info['rpm']:.1f} rpm (time unit '{info['time_unit']}')")
        for k, v in checks.items():
            print(f"  check {k}: {v:.3g}")
        if info["inductance"] == "incremental":
            st = info["inductance_stats"]
            print("Inductance columns: DERIVED incremental matrix (not Maxwell frozen-permeability).")
            print(f"  Ld {st['Ld_incremental_uH_range'][0]:.0f}..{st['Ld_incremental_uH_range'][1]:.0f} uH, "
                  f"Lq {st['Lq_incremental_uH_range'][0]:.0f}..{st['Lq_incremental_uH_range'][1]:.0f} uH, "
                  f"reciprocity |Ldq-Lqd| median {st['reciprocity_|Ldq-Lqd|_median_uH']:.2f} uH")
        else:
            print("Inductance columns left empty (--inductance none).")
        print(f"Analyse with turns_ratio = 1, stack_factor = 1, pole_pairs = {pole_pairs}.")
    return checks


def _default_locations():
    """Where to look for the Zenodo CSVs when no paths are given.

    1. Repo layout: <repo>/Scripts/create_zenodo_input_data.py with the CSVs in
       <repo>/Input_Data/Zenodo_IPM/. Output goes to the same Zenodo_IPM folder.
    2. Otherwise, the CSVs in the same folder as this script, with output beside them.
    """
    here = Path(__file__).resolve().parent
    repo_src = here.parent / "Input_Data" / "Zenodo_IPM"
    if (repo_src / "Psi_d.csv").exists():
        return repo_src
    if (here / "Psi_d.csv").exists():
        return here
    return repo_src                        # neither found; reported clearly when run


def main(argv=None):
    """Command-line entry point (see Usage in the module docstring)."""
    src = _default_locations()
    p = argparse.ArgumentParser(description="Turn the Zenodo IPM dq maps into zenodo_input_data.csv")
    p.add_argument("--psi-d", type=Path, default=src / "Psi_d.csv")
    p.add_argument("--psi-q", type=Path, default=src / "Psi_q.csv")
    p.add_argument("--torque", type=Path, default=src / "Torque.csv",
                   help="optional; used for the summary file and the torque check")
    p.add_argument("--out", type=Path, default=src / "zenodo_input_data.csv")
    p.add_argument("--pole-pairs", type=int, default=POLE_PAIRS_DEFAULT)
    p.add_argument("--conductor-area", type=float, default=1.0,
                   help="m^2, only used for the J proxy (J = Im / area)")
    p.add_argument("--time-unit", choices=sorted(_TIME_UNITS), default=None,
                   help="override the unit in the time column header")
    p.add_argument("--inductance", choices=["incremental", "none"], default="incremental",
                   help="fill the L(...) columns with the flux-derived incremental "
                        "inductance matrix (default) or leave them empty")
    args = p.parse_args(argv)
    try:
        create_zenodo_input_data(args.psi_d, args.psi_q, args.torque, args.out,
                                 args.pole_pairs, args.conductor_area, args.time_unit,
                                 args.inductance)
    except FileNotFoundError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
