"""
test_zenodo_input_data.py - checks for create_zenodo_input_data.py (the converter).

  * the inverse Park transform is undone exactly by the Park transform
  * the derived inductance matrix is symmetric and returns its dq values
  * the inductance derivation recovers known Ld, Lq and cross-coupling
  * the real conversion (if the Zenodo files are present) passes its self-checks
"""

import numpy as np
import pytest

from conftest import ZENODO_RAW
from create_zenodo_input_data import (create_zenodo_input_data, dq_to_abc_inductance,
                                      incremental_inductance_dq, inverse_park, park)


def test_inverse_park_round_trip():
    rng = np.random.default_rng(2)
    d, q, theta = rng.normal(size=50), rng.normal(size=50), rng.uniform(0, 2 * np.pi, 50)
    d2, q2 = park(*inverse_park(d, q, theta), theta)
    np.testing.assert_allclose(d2, d, atol=1e-12)
    np.testing.assert_allclose(q2, q, atol=1e-12)


def test_inductance_matrix_round_trip():
    """T @ L_abc @ inv(T) gives back Ld, Lq and the cross term; L_abc is symmetric."""
    for theta in np.linspace(0, 2 * np.pi, 7):
        L = dq_to_abc_inductance(60e-6, 120e-6, -8e-6, theta)
        c = [np.cos(theta), np.cos(theta - 2 * np.pi / 3), np.cos(theta + 2 * np.pi / 3)]
        s = [np.sin(theta), np.sin(theta - 2 * np.pi / 3), np.sin(theta + 2 * np.pi / 3)]
        T = (2 / 3) * np.array([c, [-x for x in s], [0.5] * 3])
        M = np.array([[L[(x, y)] for y in "ABC"] for x in "ABC"])
        D = T @ M @ np.linalg.inv(T)
        np.testing.assert_allclose([D[0, 0], D[1, 1], D[0, 1]], [60e-6, 120e-6, -8e-6], atol=1e-15)
        np.testing.assert_allclose(M, M.T, atol=1e-18)


def test_inductance_derivation_known_motor():
    """A linear motor's Ld, Lq and cross-coupling are recovered from its flux map
    on the Zenodo grid (0-620 A in 10 A steps, 10 deg steps)."""
    im = np.arange(0, 621, 10.0)
    g = np.arange(-180, 180, 10.0)
    G, IM = np.meshgrid(np.deg2rad(g), im)
    Id, Iq = -IM * np.sin(G), IM * np.cos(G)
    psi_d = 0.047 + 60e-6 * Id - 8e-6 * Iq
    psi_q = 120e-6 * Iq - 8e-6 * Id
    Ldd, Ldq, Lqd, Lqq = incremental_inductance_dq(im, g, psi_d[None], psi_q[None])
    np.testing.assert_allclose(Ldd, 60e-6, atol=1e-8)
    np.testing.assert_allclose(Lqq, 120e-6, atol=1e-8)
    np.testing.assert_allclose(Ldq, -8e-6, atol=1e-8)
    np.testing.assert_allclose(Lqd, -8e-6, atol=1e-8)


@pytest.mark.skipif(not (ZENODO_RAW / "Psi_d.csv").exists(), reason="Zenodo CSVs not present")
def test_real_conversion_self_checks(tmp_path):
    """Converting the real Zenodo files: lossless round trip, and the flux-formula
    torque matches the FEA torque."""
    checks = create_zenodo_input_data(ZENODO_RAW / "Psi_d.csv", ZENODO_RAW / "Psi_q.csv",
                                      ZENODO_RAW / "Torque.csv", tmp_path / "zenodo_input_data.csv",
                                      verbose=False)
    assert checks["max_err_psi_d_Wb"] < 1e-12 and checks["max_err_psi_q_Wb"] < 1e-12
    assert checks["max_err_Ld_from_matrix_uH"] < 1e-9
    assert checks["torque_flux_vs_FEA_rms_Nm"] < 0.1
