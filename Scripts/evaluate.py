"""
evaluate.py - how accurate and how fast is each flux model?

Where this fits in the project
------------------------------
This file produces the evidence for the report: the model comparison
(section 4), the validation (section 5) and the performance analysis
(section 6). Nothing in it changes a model; it only measures.

Accuracy (all errors are per-unit, so the two motors compare directly)
    flux error    = RMSE of (psi_d, psi_q) as % of psi_pm
    torque error  = RMSE of torque as % of base torque, 1.5 * p * psi_pm * I_max

    kfold_cv()          Cobalt: score each model on FEA points it never saw
    select_settings()   choose polynomial degree and network size by kfold_cv
    held_out_rings()    train without the outer current rings, test on them
                        (extrapolation)
    compare_models()    the alternatives table: all models, both tests
    zenodo_validation() the frozen script on an independent motor. Train on a
                        Cobalt-like coarse subset of the Zenodo map, then test
                        on the dense points it never saw and against the FEA's
                        own torque.
    torque_formula_comparison()       flux torque vs 07-07's matrix torque
    matrix_inductance_consistency()   do Maxwell's matrix Ld/Lq reproduce the
                                      FEA flux?

Performance
    time_preprocessing()  07-07's row-by-row loop vs the vectorised transform
    profile_pipeline()    cProfile: where preprocessing spends its time
    model_costs()         per model: fit time, prediction time, memory, FLOPs
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np
import pandas as pd

import preprocess as pp
from derive import incremental_inductance, torque


# ============================================================================
# ERROR METRICS
# ============================================================================
def _rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def _errors(psi_d_hat, psi_q_hat, T_hat, pts, base, torque_col="Torque_Nm"):
    """Flux and torque errors of predictions against the FEA points `pts`."""
    e_flux = np.r_[psi_d_hat - pts["Psi_d"].to_numpy(),
                   psi_q_hat - pts["Psi_q"].to_numpy()] / base["psi_base"] * 100
    e_T = (T_hat - pts[torque_col].to_numpy()) / base["T_base"] * 100
    return {"flux_rmse_pct": _rms(e_flux), "flux_max_pct": float(np.max(np.abs(e_flux))),
            "torque_rmse_pct": _rms(e_T), "torque_max_pct": float(np.max(np.abs(e_T)))}


def _fit(make_model, pts):
    """Fit a fresh model (make_model is e.g. lambda: polynomial_regression(3))."""
    return make_model().fit(pts["Id"], pts["Iq"], pts["Psi_d"], pts["Psi_q"])


def _predict(model, pts, pole_pairs):
    """Flux and torque predicted at the operating points in `pts`."""
    Id, Iq = pts["Id"].to_numpy(), pts["Iq"].to_numpy()
    psi_d, psi_q = model.predict_flux(Id, Iq)
    return psi_d, psi_q, torque(model, Id, Iq, pole_pairs)


# ============================================================================
# CROSS-VALIDATION AND SETTINGS
# ============================================================================
def kfold_cv(make_model, table, pole_pairs, k=5, repeats=3, seed=0):
    """k-fold cross-validation on the unique FEA points.

    The points are shuffled and split into k folds. Each fold is predicted by
    a model trained on the other k-1, so every point is scored by a model that
    never saw it. This is repeated with `repeats` different shuffles and the
    errors averaged, so one lucky split cannot decide the result.
    """
    pts = pp.unique_operating_points(table)
    base = pp.base_values(table, pole_pairs)
    n = len(pts)
    results = []
    for r in range(repeats):
        order = np.random.default_rng(seed + r).permutation(n)
        psi_d_hat, psi_q_hat, T_hat = np.empty(n), np.empty(n), np.empty(n)
        for fold in np.array_split(order, k):
            train = pts.drop(index=fold)                     # everything except this fold
            model = _fit(make_model, train)
            psi_d_hat[fold], psi_q_hat[fold], T_hat[fold] = _predict(model, pts.iloc[fold], pole_pairs)
        results.append(_errors(psi_d_hat, psi_q_hat, T_hat, pts, base))
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}


def select_settings(table, pole_pairs, degrees=(2, 3, 4, 5, 6), sizes=(8, 16, 32),
                    k=5, repeats=3, seed=0):
    """Choose polynomial degree and network size by k-fold cross-validation.

    The setting with the lowest cross-validated flux error wins. Returns
    (results table, best degree, best size). Run this on Cobalt only: the
    Zenodo motor is kept for the final independent check.
    """
    from models import neural_network, polynomial_regression
    rows = []
    for d in degrees:
        rows.append({"model": "polynomial_regression", "setting": d,
                     **kfold_cv(lambda d=d: polynomial_regression(d), table, pole_pairs, k, repeats, seed)})
    for s in sizes:
        rows.append({"model": "neural_network", "setting": s,
                     **kfold_cv(lambda s=s: neural_network(s), table, pole_pairs, k, repeats, seed)})
    res = pd.DataFrame(rows)
    best = lambda name: int(res.loc[res[res.model == name].flux_rmse_pct.idxmin(), "setting"])
    return res, best("polynomial_regression"), best("neural_network")


# ============================================================================
# EXTRAPOLATION: HELD-OUT OUTER RINGS
# ============================================================================
def held_out_rings(make_model, table, pole_pairs, n_rings=2):
    """Train without the n_rings highest-current rings, then test on them.

    The FEA sweep is a set of current-magnitude rings (the J values). Removing
    the outer rings measures how a model behaves beyond the currents it was
    trained on. For Cobalt that is the 102 A and 113 A rings.
    """
    pts = pp.unique_operating_points(table)
    base = pp.base_values(table, pole_pairs)
    levels = np.sort(pts[pp.COL_J].unique())
    held = pts[pp.COL_J] >= levels[-n_rings]
    model = _fit(make_model, pts[~held])
    test = pts[held]
    return _errors(*_predict(model, test, pole_pairs), test, base)


def compare_models(factories, table, pole_pairs, k=5, repeats=3, n_rings=2):
    """The alternatives table: every model, cross-validated (inside the data)
    and on held-out rings (outside it). `factories` = {name: make_model}."""
    rows = []
    for name, make in factories.items():
        cv = kfold_cv(make, table, pole_pairs, k, repeats)
        ring = held_out_rings(make, table, pole_pairs, n_rings)
        rows.append({"model": name,
                     **{f"cv_{key}": v for key, v in cv.items()},
                     **{f"rings_{key}": v for key, v in ring.items()}})
    return pd.DataFrame(rows)


# ============================================================================
# VALIDATION ON THE INDEPENDENT ZENODO MOTOR
# ============================================================================
def zenodo_split(table, n_train_rings=11, train_angles=(0, 20, 40, 60, 80, 90),
                 outside_angles=(-20, -10, 100, 110)):
    """Split the dense Zenodo map so it imitates Cobalt's data.

    train   : about 11 current rings x 6 angles in the first quadrant (Id <= 0,
              Iq >= 0), plus the no-load point. Roughly the density of the
              Cobalt FEA (71 points).
    inside  : every other first-quadrant point up to the highest training ring
              (interpolation)
    outside : angles just outside the training quadrant, same currents
              (extrapolation)
    Returns (unique points, train mask, inside mask, outside mask).
    """
    pts = pp.unique_operating_points(table)
    J = pts[pp.COL_J].to_numpy()
    T = pts[pp.COL_TIMING].round(6).to_numpy()
    levels = np.sort(np.unique(J))
    step = max(1, int(round((len(levels) - 1) / (n_train_rings - 1))))
    train_levels = levels[::step]
    J_top = train_levels.max()
    train = (np.isin(J, train_levels) & np.isin(T, train_angles)) | (J == 0)
    inside = ~train & (T >= 0) & (T <= 90) & (J <= J_top) & (J > 0)
    outside = np.isin(T, outside_angles) & (J <= J_top) & (J > 0)
    return pts, train, inside, outside


def zenodo_validation(factories, table, pole_pairs, **split_kwargs):
    """Run each (frozen) model on the Zenodo motor.

    Each model is trained on the coarse subset only, then tested on:
      * flux at the dense points inside and outside the training region
      * torque against the FEA's own torque (Torque_FEA_Nm), which never went
        through any of this code. That is the independent check.
      * incremental inductance against Zenodo's dense-grid reference, the
        columns Ld_matrix/Lq_matrix. For Zenodo these come from 5-point finite
        differences on the full 10 A x 10 deg grid (create_zenodo_input_data.py),
        not from Maxwell.
    """
    pts, train, inside, outside = zenodo_split(table, **split_kwargs)
    base = pp.base_values(table, pole_pairs)
    torque_col = "Torque_FEA_Nm" if "Torque_FEA_Nm" in pts else "Torque_Nm"
    rows = []
    for name, make in factories.items():
        model = _fit(make, pts[train])
        row = {"model": name, "n_train": int(train.sum()),
               "n_inside": int(inside.sum()), "n_outside": int(outside.sum())}
        for label, mask in (("inside", inside), ("outside", outside)):
            sub = pts[mask]
            err = _errors(*_predict(model, sub, pole_pairs), sub, base, torque_col)
            row.update({f"{label}_{key}": v for key, v in err.items()})
        # Incremental inductance inside the training region, vs the dense reference
        sub = pts[inside]
        inc = incremental_inductance(model, sub["Id"].to_numpy(), sub["Iq"].to_numpy())
        for out_key, ref_col in (("Ld", "Ld_matrix"), ("Lq", "Lq_matrix")):
            ref = sub[ref_col].to_numpy()
            rel = (inc[out_key] - ref) / np.mean(np.abs(ref)) * 100
            row[f"inside_{out_key}_rmse_pct"] = _rms(rel)
            # The median is not dominated by a few points: the lookup table's
            # slope jumps between triangles and at its nearest-point fallback
            row[f"inside_{out_key}_median_abs_pct"] = float(np.median(np.abs(rel)))
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# FORMULA CHECKS
# ============================================================================
def torque_formula_comparison(table, pole_pairs):
    """Flux-formula torque vs 07-07's matrix-inductance formula.

    With FEA torque available (Zenodo), both are scored against it. Without it
    (Cobalt), the two formulas are compared with each other.
    """
    pts = pp.unique_operating_points(table)
    base = pp.base_values(table, pole_pairs)
    rows = []
    ref_col = "Torque_FEA_Nm" if "Torque_FEA_Nm" in pts else "Torque_Nm"
    for col, label in (("Torque_Nm", "flux formula"), ("Torque_matrix_Nm", "07-07 matrix formula")):
        e = pts[col] - pts[ref_col]
        rows.append({"formula": label, "reference": ref_col,
                     "rmse_Nm": _rms(e), "max_abs_Nm": float(e.abs().max()),
                     "rmse_pct_of_base": _rms(e) / base["T_base"] * 100,
                     "peak_torque_Nm": float(pts[col].max())})
    return pd.DataFrame(rows)


def matrix_inductance_consistency(table):
    """Does psi = psi_pm_eff + L * i reproduce the FEA flux?

    Checked with the Maxwell matrix inductances and with the apparent ones.
    Apparent inductances reproduce the flux by definition (a sanity check).
    Errors are % of psi_pm.
    """
    pts = pp.unique_operating_points(table)
    psi_pm = pts["Psi_pm"].iloc[0]
    rows = []
    for label, Ld, Lq in (("matrix", "Ld_matrix", "Lq_matrix"), ("apparent", "Ld_app", "Lq_app")):
        e_d = (pts["Psi_pm_eff"] + pts[Ld] * pts["Id"] - pts["Psi_d"]) / psi_pm * 100
        e_q = (pts[Lq] * pts["Iq"] - pts["Psi_q"]) / psi_pm * 100
        rows.append({"inductances": label,
                     "psi_d_rmse_pct": _rms(e_d), "psi_d_max_pct": float(e_d.abs().max()),
                     "psi_q_rmse_pct": _rms(e_q), "psi_q_max_pct": float(e_q.abs().max())})
    return pd.DataFrame(rows)


# ============================================================================
# PERFORMANCE
# ============================================================================
def time_preprocessing(path, settings, repeats=3):
    """Time the Park transform of the inductance matrix: 07-07's row-by-row
    loop vs the vectorised version (best of `repeats` runs each). Also checks
    that both give the same numbers."""
    raw = pp.load_winding_table(path)
    data = pp.rotor_angle(raw, verbose=False)
    rows, results = [], {}
    for method in ("loop", "vectorised"):
        best = np.inf
        for _ in range(repeats):
            d = data.copy()
            t0 = time.perf_counter()
            d = pp.dq_per_row(d, method)
            best = min(best, time.perf_counter() - t0)
        results[method] = d[["Ldm_raw", "Lqm_raw"]].to_numpy()
        rows.append({"method": method, "rows": len(data), "seconds": best})
    df = pd.DataFrame(rows)
    df["speed_up"] = df.loc[df.method == "loop", "seconds"].iloc[0] / df["seconds"]
    df["max_difference_H"] = float(np.max(np.abs(results["loop"] - results["vectorised"])))
    return df


def profile_pipeline(path, settings, top=12):
    """cProfile of the whole preprocessing stage: the `top` functions by
    cumulative time. Shows where the time goes, so optimisation effort goes there."""
    profiler = cProfile.Profile()
    profiler.enable()
    pp.build_flux_map(path, verbose=False, **settings)
    profiler.disable()
    stats = pstats.Stats(profiler).stats            # {(file, line, function): (cc, nc, tt, ct, callers)}
    rows = [{"function": f"{fn} ({file.split('/')[-1].split(chr(92))[-1]}:{line})",
             "calls": nc, "own_time_s": tt, "cumulative_time_s": ct}
            for (file, line, fn), (cc, nc, tt, ct, callers) in stats.items()]
    return (pd.DataFrame(rows).sort_values("cumulative_time_s", ascending=False)
              .head(top).reset_index(drop=True))


def model_costs(factories, table, n_queries=100_000, repeats=3, seed=0):
    """Per model: fit time, prediction time for a million queries, stored
    parameters (memory) and estimated arithmetic per prediction (FLOPs).
    Queries are random currents inside the training range."""
    pts = pp.unique_operating_points(table)
    rng = np.random.default_rng(seed)
    rows = []
    for name, make in factories.items():
        fit_best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            model = _fit(make, pts)
            fit_best = min(fit_best, time.perf_counter() - t0)
        # Random query points inside the FEA range (polar: magnitude x angle)
        lo, hi = model.beta_range
        I = model.I_base * np.sqrt(rng.random(n_queries))
        beta = np.deg2rad(lo + (hi - lo) * rng.random(n_queries))
        Id, Iq = -I * np.sin(beta), I * np.cos(beta)
        pred_best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.predict_flux(Id, Iq)
            pred_best = min(pred_best, time.perf_counter() - t0)
        rows.append({"model": name, "fit_ms": fit_best * 1e3,
                     "predict_s_per_million": pred_best * 1e6 / n_queries,
                     "parameters": model.n_parameters,
                     "memory_kB": model.n_parameters * 8 / 1024,
                     "flops_per_eval": model.flops_per_eval()})
    return pd.DataFrame(rows)
