"""
test_preprocess.py - checks for preprocess.py.

  * preprocess_v3's row loop and the vectorised transform give the same numbers
  * the full preprocessing recovers the known-answer motor's flux and inductances
  * the Park transform and its inverse undo each other
  * bad input files are rejected with a clear error
"""

import numpy as np
import pandas as pd
import pytest

import preprocess as pp
from conftest import known_flux, known_inductance, PSI_PM


def test_loop_and_vectorised_agree(known_winding_table):
    """preprocess_v3's iterrows loop and the vectorised version compute the same Ld, Lq."""
    a = pp.build_flux_map(known_winding_table, 1, 1, 4, inductance_method="loop", verbose=False)
    b = pp.build_flux_map(known_winding_table, 1, 1, 4, inductance_method="vectorised", verbose=False)
    np.testing.assert_allclose(a["Ld_matrix"], b["Ld_matrix"], rtol=1e-12)
    np.testing.assert_allclose(a["Lq_matrix"], b["Lq_matrix"], rtol=1e-12)


def test_recovers_known_motor(known_winding_table):
    """Known-answer check: preprocessing a winding table of a motor whose flux is
    known exactly must give that flux back, and sensible inductances."""
    m = pp.build_flux_map(known_winding_table, 1, 1, 4, verbose=False)
    u = pp.unique_operating_points(m)
    psi_d, psi_q = known_flux(u["Id"], u["Iq"])
    np.testing.assert_allclose(u["Psi_d"], psi_d, atol=1e-9)
    np.testing.assert_allclose(u["Psi_q"], psi_q, atol=1e-9)
    assert abs(u["Psi_pm"].iloc[0] - PSI_PM) < 1e-9

    exact = known_inductance(u["Id"], u["Iq"])
    # Matrix columns: converter's 5-point stencil on the 10 A x 10 deg grid -> very accurate
    np.testing.assert_allclose(u["Ld_matrix"], exact["Ld"], rtol=1e-3)
    np.testing.assert_allclose(u["Lq_matrix"], exact["Lq"], rtol=1e-3)
    # preprocess_v3's grid finite differences (3-point stencil): within 1 % on the interior
    # rings (the stencil's bias on a 10 deg grid is about 0.5 %). The outermost ring
    # is excluded: np.gradient is one-sided there, about 2 % off on this motor.
    ok = u["Ld_inc"].notna() & (u[pp.COL_J] < u[pp.COL_J].max())
    np.testing.assert_allclose(u.loc[ok, "Ld_inc"], exact["Ld"][ok], rtol=1e-2)
    np.testing.assert_allclose(u.loc[ok, "Lq_inc"], exact["Lq"][ok], rtol=1e-2)


def test_park_round_trip():
    """Park transform of the inverse Park transform returns the original dq values."""
    from create_zenodo_input_data import inverse_park
    rng = np.random.default_rng(1)
    d, q, theta = rng.normal(size=100), rng.normal(size=100), rng.uniform(0, 2 * np.pi, 100)
    a, b, c = inverse_park(d, q, theta)
    d2, q2, z = pp.park_vec(a, b, c, theta)
    np.testing.assert_allclose(d2, d, atol=1e-12)
    np.testing.assert_allclose(q2, q, atol=1e-12)
    np.testing.assert_allclose(z, 0, atol=1e-12)


def test_bad_inputs_raise(known_winding_table, tmp_path):
    """Unusable input files stop with a clear error rather than a wrong answer."""
    good = pd.read_csv(known_winding_table)

    with pytest.raises(FileNotFoundError):
        pp.load_winding_table(tmp_path / "does_not_exist.csv")

    def check(df, name):
        path = tmp_path / name
        df.to_csv(path, index=False)
        with pytest.raises(ValueError):
            pp.load_winding_table(path)

    check(good.drop(columns=["FluxLinkage(Phase_A) [Wb]"]), "missing_column.csv")
    check(good.iloc[:-1], "not_whole_blocks.csv")
    no_load = good.copy()
    no_load.loc[no_load[pp.COL_J] == 0, pp.COL_J] = 1.0
    check(no_load, "no_no_load.csv")
    with_nan = good.copy()
    with_nan.loc[5, "InputCurrent(Phase_A) [mA]"] = np.nan
    check(with_nan, "nan_current.csv")
    unbalanced = good.copy()
    unbalanced["InputCurrent(Phase_A) [mA]"] += 1000.0
    check(unbalanced, "unbalanced.csv")
