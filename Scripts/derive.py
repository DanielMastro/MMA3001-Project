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

__docformat__ = "numpy"   # docstring style, read by pdoc

import numpy as np
import pandas as pd


# ============================================================================
# TORQUE
# ============================================================================
def torque_from_flux(psi_d, psi_q, i_d, i_q, pole_pairs):
    """Compute electromagnetic torque from dq flux linkage and current.

    Uses T = 1.5 * p * (psi_d * i_q - psi_q * i_d). The 1.5 factor belongs to
    the amplitude-invariant (peak) dq convention used everywhere in this
    project. No inductance is involved, so the torque is exact for whatever
    flux it is given.

    Parameters
    ----------
    psi_d : float or array_like
        d-axis flux linkage [Wb].
    psi_q : float or array_like
        q-axis flux linkage [Wb].
    i_d : float or array_like
        d-axis current [A], peak value.
    i_q : float or array_like
        q-axis current [A], peak value.
    pole_pairs : int
        Number of pole pairs p.

    Returns
    -------
    torque : numpy.ndarray or float
        Torque [Nm], broadcast to the shape of the inputs.

    See Also
    --------
    matrix_torque : The original script's inductance-based formula.

    Examples
    --------
    >>> float(torque_from_flux(0.05, 0.0, 0.0, 100.0, pole_pairs=5))
    37.5
    """
    return 1.5 * pole_pairs * (np.asarray(psi_d) * np.asarray(i_q)
                               - np.asarray(psi_q) * np.asarray(i_d))


def matrix_torque(psi_pm_eff, Ld, Lq, i_d, i_q, pole_pairs):
    """Compute torque with the original script's inductance formula.

    Uses T = 1.5 * p * (psi_pm_eff + (Ld - Lq) * i_d) * i_q, the formula in
    History/preprocess_v3.py. It is kept only as the alternative that
    torque_from_flux() is compared against. It equals the flux formula only
    when Ld and Lq are apparent inductances that reproduce the FEA flux.
    Maxwell's matrix inductances do not, so with them this formula drifts
    from the FEA torque at high load.

    Parameters
    ----------
    psi_pm_eff : float or array_like
        Effective magnet flux linkage at the operating point's Iq [Wb].
    Ld : float or array_like
        d-axis inductance [H].
    Lq : float or array_like
        q-axis inductance [H].
    i_d : float or array_like
        d-axis current [A], peak value.
    i_q : float or array_like
        q-axis current [A], peak value.
    pole_pairs : int
        Number of pole pairs p.

    Returns
    -------
    torque : numpy.ndarray or float
        Torque [Nm], broadcast to the shape of the inputs.

    See Also
    --------
    torque_from_flux : The flux formula used for every reported torque.
    """
    return 1.5 * pole_pairs * (np.asarray(psi_pm_eff)
                               + (np.asarray(Ld) - np.asarray(Lq)) * np.asarray(i_d)) * np.asarray(i_q)


def torque(model, Id, Iq, pole_pairs):
    """Predict torque at any operating point from a fitted flux model.

    The model's predicted flux is passed through torque_from_flux().

    Parameters
    ----------
    model : flux model
        A fitted model from models.py (anything with predict_flux).
    Id : float or array_like
        d-axis current [A], peak value.
    Iq : float or array_like
        q-axis current [A], peak value.
    pole_pairs : int
        Number of pole pairs p.

    Returns
    -------
    torque : numpy.ndarray
        Torque [Nm], the broadcast shape of Id and Iq.
    """
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
    """Compute the incremental (differential) inductances at (Id, Iq).

    These are the slopes of the flux map, the inductances the inverter's
    current controller sees. With method="analytic" they come from the
    model's own exact gradient (flux_gradient). With "fd3" or "fd5" a
    finite-difference stencil is applied to the model's predict_flux with
    step h. Their errors fall as h^2 and h^4; they are the alternatives
    compared in the report.

    Parameters
    ----------
    model : flux model
        A fitted model from models.py.
    Id : float or array_like
        d-axis current [A], peak value.
    Iq : float or array_like
        q-axis current [A], peak value.
    method : {"analytic", "fd3", "fd5"}, optional
        How the derivative is taken. Default "analytic".
    h : float, optional
        Finite-difference step [A] for "fd3" and "fd5". Default 1 % of the
        largest training current (model.I_base). Ignored for "analytic".

    Returns
    -------
    inductances : dict of numpy.ndarray
        Inductances [H], each the shape of Id and Iq: "Ld" = dpsi_d/dId,
        "Lq" = dpsi_q/dIq, and the cross-coupling terms "Ldq" = dpsi_d/dIq
        and "Lqd" = dpsi_q/dId (equal for a physically consistent map).

    Raises
    ------
    ValueError
        If method is not "analytic", "fd3" or "fd5".
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
    """Compute the apparent (secant) inductances at (Id, Iq).

    These are the inductances that satisfy psi = psi_pm + L * i exactly:

        Ld = (psi_d(Id, Iq) - psi_pm_eff(Iq)) / Id,  with psi_pm_eff(Iq) = psi_d(0, Iq)
        Lq = psi_q(Id, Iq) / Iq

    psi_pm_eff(Iq) is psi_d along the Id = 0 line, the same definition the
    original script used. It falls with |Iq| because of cross-saturation.
    Near the axes the ratios are 0/0, so where |Id| (or |Iq|) is below
    max(rel_tol * |I|, abs_tol) the limiting value, the incremental
    inductance, is used instead.

    Parameters
    ----------
    model : flux model
        A fitted model from models.py.
    Id : float or array_like
        d-axis current [A], peak value.
    Iq : float or array_like
        q-axis current [A], peak value.
    rel_tol : float, optional
        Relative threshold for the 0/0 fallback, as a fraction of |I|.
        Default 0.02, the original script's value.
    abs_tol : float, optional
        Absolute threshold for the 0/0 fallback [A]. Default 0.5, the
        original script's value.

    Returns
    -------
    inductances : dict of numpy.ndarray
        "Ld" and "Lq" [H] and "Psi_pm_eff" [Wb], each the shape of Id and Iq.
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
    """Make a dense grid of operating points in current magnitude and angle.

    The layout matches the FEA sweeps (magnitude rings x angles), only finer,
    with Id = -|I| sin(beta) and Iq = |I| cos(beta). The zero-current point
    appears once.

    Parameters
    ----------
    I_max : float
        Largest current magnitude [A].
    beta_min : float, optional
        First current angle [deg]. Default 0 (pure +q).
    beta_max : float, optional
        Last current angle [deg]. Default 90 (pure -d).
    dI : float, optional
        Step in current magnitude [A]. Default 1.
    dbeta : float, optional
        Step in current angle [deg]. Default 1.

    Returns
    -------
    Id : numpy.ndarray
        d-axis current of every grid point [A], as a flat array.
    Iq : numpy.ndarray
        q-axis current of every grid point [A], the same length.

    Examples
    --------
    >>> Id, Iq = polar_grid(113.0)
    >>> Id.shape
    (10284,)
    """
    I = np.linspace(0.0, I_max, int(round(I_max / dI)) + 1)
    beta = np.deg2rad(np.linspace(beta_min, beta_max, int(round((beta_max - beta_min) / dbeta)) + 1))
    II, BB = np.meshgrid(I[1:], beta, indexing="ij")            # rings above zero current
    Id = np.r_[0.0, (-II * np.sin(BB)).ravel()]
    Iq = np.r_[0.0, (II * np.cos(BB)).ravel()]
    return Id, Iq


def output_map(model, Id, Iq, pole_pairs):
    """Tabulate every output at the given operating points.

    One row per operating point with flux, torque, both kinds of inductance
    and an extrapolation flag. run_all.py writes this table for Cobalt to
    Output_Data/Cobalt/cobalt_output_map.csv.

    Parameters
    ----------
    model : flux model
        A fitted model from models.py.
    Id : array_like
        d-axis currents [A], peak values.
    Iq : array_like
        q-axis currents [A], peak values, the same length as Id.
    pole_pairs : int
        Number of pole pairs p.

    Returns
    -------
    table : pandas.DataFrame
        One row per operating point, with columns Id and Iq [A]; Psi_d,
        Psi_q, Psi_mag and Psi_pm_eff [Wb]; Torque_Nm; Ld_app_uH, Lq_app_uH,
        Ld_inc_uH and Lq_inc_uH [uH]; Ldq_inc_uH, the cross-coupling (mean of
        dpsi_d/dIq and dpsi_q/dId) [uH]; Saliency (apparent Lq/Ld); and
        Extrapolated (True outside the FEA range, where results should be
        treated with care).

    Examples
    --------
    >>> import preprocess as pp, models as md
    >>> flux_map = pp.build_flux_map("Input_Data/Cobalt/cobalt_input_data.csv",
    ...                              turns_ratio=33 / 2, stack_factor=0.9,
    ...                              pole_pairs=5, verbose=False)
    >>> pts = pp.unique_operating_points(flux_map)
    >>> model = md.polynomial_regression(6).fit(pts["Id"], pts["Iq"],
    ...                                         pts["Psi_d"], pts["Psi_q"])
    >>> out = output_map(model, Id=[-20.0], Iq=[60.0], pole_pairs=5)
    >>> print(f"{out['Torque_Nm'].iloc[0]:.1f} Nm, extrapolated: {out['Extrapolated'].iloc[0]}")
    20.6 Nm, extrapolated: False
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
