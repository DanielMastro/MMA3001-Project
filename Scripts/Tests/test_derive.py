"""
test_derive.py - checks for derive.py (the known-answer verification).

The known-answer motor in conftest.py has exact formulas for flux and
inductance, so derive.py's results can be compared with the true values:
  * the 5-point stencil recovers the true incremental inductances
  * the stencils converge at their theoretical rates (h^2 and h^4)
  * a polynomial model fitted to a polynomial motor gives exact inductances
  * apparent inductance and torque match their definitions
"""

import numpy as np

import derive as dv
from conftest import known_inductance
from models import polynomial_regression


def test_torque_formula():
    """T = 1.5 * p * (psi_d * iq - psi_q * id), checked by hand."""
    assert np.isclose(dv.torque_from_flux(0.05, 0.02, -10.0, 100.0, 5), 1.5 * 5 * (0.05 * 100 + 0.02 * 10))


def test_fd5_recovers_known_inductance(known_motor):
    """The 5-point stencil gives the true incremental inductances."""
    Id = np.linspace(-100, 0, 11)
    Iq = np.linspace(0, 100, 11)
    est = dv.incremental_inductance(known_motor, Id, Iq, method="fd5", h=1.0)
    exact = known_inductance(Id, Iq)
    for key in ("Ld", "Lq", "Ldq", "Lqd"):
        np.testing.assert_allclose(est[key], exact[key], rtol=1e-6, err_msg=key)


def test_stencils_converge_at_their_order(known_motor):
    """Halving the step divides the 3-point error by ~4 (h^2) and the 5-point
    error by ~16 (h^4). This is the convergence study in the report."""
    Id, Iq = np.array([-80.0]), np.array([30.0])
    exact = known_inductance(Id, Iq)["Ld"]
    # Steps small enough to be in the asymptotic range (at h = 40 A the 5-point
    # ratio is still ~20), large enough that round-off does not dominate
    err = {m: [abs(dv.incremental_inductance(known_motor, Id, Iq, m, h)["Ld"] - exact)[0]
               for h in (10.0, 5.0)] for m in ("fd3", "fd5")}
    assert 3.8 < err["fd3"][0] / err["fd3"][1] < 4.2
    assert 15 < err["fd5"][0] / err["fd5"][1] < 17.5
    assert err["fd5"][1] < err["fd3"][1]


def test_polynomial_model_gives_exact_inductance():
    """Fit polynomial_regression(3) to a cubic motor; its analytic gradient must
    equal the true derivative everywhere, not only at the training points."""
    def flux(Id, Iq):
        psi_d = 0.048 + 500e-6 * Id - 20e-6 * Iq + 1e-6 * Id**2 + 2e-9 * Id**3
        psi_q = 600e-6 * Iq - 20e-6 * Id - 1e-8 * Iq**3
        return psi_d, psi_q
    I, beta = np.meshgrid(np.linspace(0, 113, 11), np.deg2rad(np.linspace(0, 90, 7)))
    Id, Iq = (-I * np.sin(beta)).ravel(), (I * np.cos(beta)).ravel()
    model = polynomial_regression(3).fit(Id, Iq, *flux(Id, Iq))

    Id_t, Iq_t = np.array([-37.0, -5.0, -90.0]), np.array([61.0, 100.0, 12.0])   # off-grid points
    inc = dv.incremental_inductance(model, Id_t, Iq_t, method="analytic")
    np.testing.assert_allclose(inc["Ld"], 500e-6 + 2e-6 * Id_t + 6e-9 * Id_t**2, rtol=1e-8)
    np.testing.assert_allclose(inc["Lq"], 600e-6 - 3e-8 * Iq_t**2, rtol=1e-8)
    np.testing.assert_allclose(inc["Ldq"], -20e-6, rtol=1e-6)


def test_apparent_inductance_of_linear_motor():
    """For psi_d = psi_pm + Ld*Id, psi_q = Lq*Iq, apparent inductance equals Ld,
    Lq everywhere, including on the axes, where the 0/0 limit is used."""
    class Linear:
        I_base = 100.0
        def predict_flux(self, Id, Iq):
            return 0.048 + 500e-6 * np.asarray(Id), 600e-6 * np.asarray(Iq)
        def flux_gradient(self, Id, Iq):
            one = np.ones(np.shape(Id))
            return {"dpsid_did": 500e-6 * one, "dpsiq_diq": 600e-6 * one,
                    "dpsid_diq": 0 * one, "dpsiq_did": 0 * one}
    Id = np.array([0.0, -50.0, -100.0, 0.0])
    Iq = np.array([100.0, 50.0, 0.0, 0.0])
    app = dv.apparent_inductance(Linear(), Id, Iq)
    np.testing.assert_allclose(app["Ld"], 500e-6)
    np.testing.assert_allclose(app["Lq"], 600e-6)
