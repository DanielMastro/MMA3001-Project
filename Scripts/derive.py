"""
derive.py - torque and inductances from a flux model.

Where this fits in the project
------------------------------
preprocess.py gives flux at the FEA operating points, and models.py fits a flux
model so flux can be predicted at ANY current. This file turns that predicted
flux into the quantities the inverter needs:

    torque                  T  = 1.5 * p * (psi_d * Iq - psi_q * Id)
    apparent inductance     Ld = (psi_d - psi_pm_eff(Iq)) / Id,   Lq = psi_q / Iq
    incremental inductance  Ld = dpsi_d/dId,  Lq = dpsi_q/dIq,  Ldq = dpsi_d/dIq

Torque and apparent inductance are formulas applied to the predicted flux, so
they are only as accurate as the flux model. Incremental inductance needs a
DERIVATIVE, which is the numerical-methods part of this file. It is computed in
one of three ways, which the report compares:

    method="analytic"  the flux model's own exact gradient (models.flux_gradient)
    method="fd3"       3-point central difference on predict_flux, error ~ h^2
    method="fd5"       5-point central difference on predict_flux, error ~ h^4

Conventions (same as preprocess.py): amplitude-invariant (peak) dq quantities,
current angle beta = atan2(-Id, Iq) in degrees (0 = pure +q, 90 = pure -d).
Units: A, Wb, H, Nm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# TORQUE
# ============================================================================
def torque_from_flux(psi_d, psi_q, i_d, i_q, pole_pairs):
    """Electromagnetic torque from dq flux linkage and current.

    T = 1.5 * p * (psi_d * iq - psi_q * id)

    The 1.5 factor belongs to the amplitude-invariant (peak) dq convention used
    everywhere in this project. Works on scalars or numpy arrays.
    """
    return 1.5 * pole_pairs * (np.asarray(psi_d) * np.asarray(i_q)
                               - np.asarray(psi_q) * np.asarray(i_d))


def matrix_torque(psi_pm_eff, Ld, Lq, i_d, i_q, pole_pairs):
    """07-07's torque formula: T = 1.5 * p * (psi_pm_eff + (Ld - Lq) * Id) * Iq.

    Kept only as the ALTERNATIVE that torque_from_flux() is compared against.
    It equals the flux formula only when Ld and Lq are apparent inductances that
    reproduce the FEA flux. The Maxwell matrix inductances are not, so this
    formula drifts from the FEA torque at high load.
    """
    return 1.5 * pole_pairs * (np.asarray(psi_pm_eff)
                               + (np.asarray(Ld) - np.asarray(Lq)) * np.asarray(i_d)) * np.asarray(i_q)


def torque(model, Id, Iq, pole_pairs):
    """Torque at any (Id, Iq): the flux model's predicted flux put through
    torque_from_flux()."""
    psi_d, psi_q = model.predict_flux(Id, Iq)
    return torque_from_flux(psi_d, psi_q, Id, Iq, pole_pairs)


# ============================================================================
# INCREMENTAL INDUCTANCE (the derivative)
# ============================================================================
# Finite-difference weights for d f / dx at x, using points x + k*h.
#   3-point: [f(x+h) - f(x-h)] / (2h)                         error ~ h^2
#   5-point: [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)  error ~ h^4
_STENCILS = {
    "fd3": {1: 1 / 2, -1: -1 / 2},
    "fd5": {2: -1 / 12, 1: 8 / 12, -1: -8 / 12, -2: 1 / 12},
}


def incremental_inductance(model, Id, Iq, method="analytic", h=None):
    """Incremental (differential) inductances at (Id, Iq), in henries.

    Returns a dict:
        "Ld"  = dpsi_d/dId    "Lq"  = dpsi_q/dIq
        "Ldq" = dpsi_d/dIq    "Lqd" = dpsi_q/dId   (cross-coupling; equal for
                                                     a physically consistent map)

    method = "analytic" uses model.flux_gradient (exact for the fitted model).
    method = "fd3" / "fd5" applies a finite-difference stencil to
    model.predict_flux with step h amps (default 1 % of the largest training
    current). These are the alternatives compared in the report.
    """
    Id = np.asarray(Id, dtype=float)
    Iq = np.asarray(Iq, dtype=float)

    if method == "analytic":
        g = model.flux_gradient(Id, Iq)
        return {"Ld": g["dpsid_did"], "Lq": g["dpsiq_diq"],
                "Ldq": g["dpsid_diq"], "Lqd": g["dpsiq_did"]}

    if method not in _STENCILS:
        raise ValueError(f"method must be 'analytic', 'fd3' or 'fd5', not '{method}'")
    if h is None:
        h = 0.01 * model.I_base                     # step size in amps

    # Accumulate the weighted flux values for a step in Id and a step in Iq
    dpsid_did = dpsiq_did = dpsid_diq = dpsiq_diq = 0.0
    for k, w in _STENCILS[method].items():
        pd_d, pq_d = model.predict_flux(Id + k * h, Iq)    # step along Id
        pd_q, pq_q = model.predict_flux(Id, Iq + k * h)    # step along Iq
        dpsid_did = dpsid_did + w * pd_d
        dpsiq_did = dpsiq_did + w * pq_d
        dpsid_diq = dpsid_diq + w * pd_q
        dpsiq_diq = dpsiq_diq + w * pq_q
    return {"Ld": dpsid_did / h, "Lq": dpsiq_diq / h,
            "Ldq": dpsid_diq / h, "Lqd": dpsiq_did / h}


# ============================================================================
# APPARENT INDUCTANCE
# ============================================================================
def apparent_inductance(model, Id, Iq, rel_tol=0.02, abs_tol=0.5):
    """Apparent (secant) inductances, the ones that satisfy psi = psi_pm + L * i.

        Ld = (psi_d(Id, Iq) - psi_pm_eff(Iq)) / Id     with psi_pm_eff(Iq) = psi_d(0, Iq)
        Lq =  psi_q(Id, Iq) / Iq

    psi_pm_eff(Iq) is psi_d along the Id = 0 line, the same definition 07-07 used
    (it falls with |Iq| because of cross-saturation).

    Near the axes these are 0/0, so where |Id| (or |Iq|) is below
    max(rel_tol * |I|, abs_tol) the limiting value, the incremental inductance,
    is used instead. The 2 % and 0.5 A thresholds are 07-07's.

    Returns a dict with "Ld", "Lq" (H) and "Psi_pm_eff" (Wb).
    """
    Id = np.asarray(Id, dtype=float)
    Iq = np.asarray(Iq, dtype=float)
    psi_d, psi_q = model.predict_flux(Id, Iq)
    psi_pm_eff, _ = model.predict_flux(np.zeros_like(Id), Iq)   # flux on the Id = 0 line

    I_tot = np.hypot(Id, Iq)
    ok_d = (np.abs(Id) > rel_tol * I_tot) & (np.abs(Id) > abs_tol)
    ok_q = (np.abs(Iq) > rel_tol * I_tot) & (np.abs(Iq) > abs_tol)
    inc = incremental_inductance(model, Id, Iq)                 # limits for the 0/0 points

    with np.errstate(divide="ignore", invalid="ignore"):
        Ld = np.where(ok_d, (psi_d - psi_pm_eff) / Id, inc["Ld"])
        Lq = np.where(ok_q, psi_q / Iq, inc["Lq"])
    return {"Ld": Ld, "Lq": Lq, "Psi_pm_eff": psi_pm_eff}


# ============================================================================
# OUTPUT MAPS (what gets handed to the inverter team)
# ============================================================================
def polar_grid(I_max, beta_min=0.0, beta_max=90.0, dI=1.0, dbeta=1.0):
    """A dense grid of operating points in current magnitude x current angle.

    Same layout as the FEA sweeps (magnitude rings x angles), just much finer.
    Id = -|I| sin(beta), Iq = |I| cos(beta). The zero-current point appears once.
    Returns (Id, Iq) as flat arrays.
    """
    I = np.linspace(0.0, I_max, int(round(I_max / dI)) + 1)
    beta = np.deg2rad(np.linspace(beta_min, beta_max, int(round((beta_max - beta_min) / dbeta)) + 1))
    II, BB = np.meshgrid(I[1:], beta, indexing="ij")            # rings above zero current
    Id = np.r_[0.0, (-II * np.sin(BB)).ravel()]
    Iq = np.r_[0.0, (II * np.cos(BB)).ravel()]
    return Id, Iq


def output_map(model, Id, Iq, pole_pairs):
    """Every output at the given operating points, as one table.

    Columns: Id, Iq [A]; Psi_d, Psi_q, Psi_mag, Psi_pm_eff [Wb]; Torque_Nm;
    apparent and incremental Ld, Lq and the cross-coupling Ldq [uH];
    Saliency (apparent Lq/Ld); Extrapolated (True = outside the FEA range,
    treat with care).
    """
    Id = np.asarray(Id, dtype=float)
    Iq = np.asarray(Iq, dtype=float)
    psi_d, psi_q = model.predict_flux(Id, Iq)
    app = apparent_inductance(model, Id, Iq)
    inc = incremental_inductance(model, Id, Iq)
    return pd.DataFrame({
        "Id": Id,
        "Iq": Iq,
        "Psi_d": psi_d,
        "Psi_q": psi_q,
        "Psi_mag": np.hypot(psi_d, psi_q),
        "Psi_pm_eff": app["Psi_pm_eff"],
        "Torque_Nm": torque_from_flux(psi_d, psi_q, Id, Iq, pole_pairs),
        "Ld_app_uH": app["Ld"] * 1e6,
        "Lq_app_uH": app["Lq"] * 1e6,
        "Ld_inc_uH": inc["Ld"] * 1e6,
        "Lq_inc_uH": inc["Lq"] * 1e6,
        "Ldq_inc_uH": 0.5 * (inc["Ldq"] + inc["Lqd"]) * 1e6,
        "Saliency": app["Lq"] / app["Ld"],
        "Extrapolated": model.is_extrapolation(Id, Iq),
    })
