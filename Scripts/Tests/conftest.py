"""
conftest.py - shared test data for the pytest tests in this folder.

pytest loads this file automatically. It holds:
  * the path to the raw Zenodo files (the test that needs them is skipped if
    they are missing)
  * the KNOWN-ANSWER MOTOR: a motor written down as formulas, so its exact
    flux, inductances and torque are known. The code must recover those values.
    This is the "known ground truth" verification in the report.
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ZENODO_RAW = ROOT / "Input_Data" / "Zenodo_IPM"

# ---------------------------------------------------------------------------
# Known-answer motor. Defined through a co-energy function
#   W' = PSI_PM*Id + LD0*IS^2*ln(cosh(Id/IS)) + LQ*Iq^2/2 + LC*Id*Iq
# so psi_d = dW'/dId and psi_q = dW'/dIq. That makes it physically consistent
# (dpsi_d/dIq = dpsi_q/dId). The d-axis saturates (tanh); values are Cobalt-like.
# ---------------------------------------------------------------------------
PSI_PM, LD0, LQ, LC, IS = 0.048, 500e-6, 600e-6, -20e-6, 200.0


def known_flux(Id, Iq):
    """Exact (psi_d, psi_q) of the known-answer motor [Wb]."""
    Id, Iq = np.asarray(Id, dtype=float), np.asarray(Iq, dtype=float)
    psi_d = PSI_PM + LD0 * IS * np.tanh(Id / IS) + LC * Iq
    psi_q = LQ * Iq + LC * Id
    return psi_d, psi_q


def known_inductance(Id, Iq):
    """Exact incremental inductances of the known-answer motor [H]."""
    Id = np.asarray(Id, dtype=float)
    return {"Ld": LD0 / np.cosh(Id / IS) ** 2, "Lq": LQ + 0 * Id,
            "Ldq": LC + 0 * Id, "Lqd": LC + 0 * Id}


class KnownMotor:
    """The known-answer motor with the same methods as a fitted flux model, so
    derive.py functions can be tested on it directly."""
    I_base = 100.0

    def predict_flux(self, Id, Iq):
        return known_flux(Id, Iq)

    def flux_gradient(self, Id, Iq):
        L = known_inductance(Id, Iq)
        return {"dpsid_did": L["Ld"], "dpsiq_diq": L["Lq"],
                "dpsid_diq": L["Ldq"], "dpsiq_did": L["Lqd"]}

    def is_extrapolation(self, Id, Iq):
        return np.zeros(np.shape(Id), dtype=bool)


@pytest.fixture
def known_motor():
    return KnownMotor()


@pytest.fixture
def known_winding_table(tmp_path):
    """The known-answer motor written as a winding table (Maxwell format),
    using the Zenodo converter's own code: 0-100 A in 10 A rings x a full
    circle of angles in 10 deg steps, 61 rotor positions per point.
    Returns the CSV path; analyse with turns_ratio=1, stack_factor=1."""
    from create_zenodo_input_data import build_winding_table

    gamma_grid, im_grid = np.meshgrid(np.arange(-180, 181, 10.0), np.arange(0, 101, 10.0))
    gamma, im = gamma_grid.ravel(), im_grid.ravel()
    Id = -im * np.sin(np.deg2rad(gamma))
    Iq = im * np.cos(np.deg2rad(gamma))
    psi_d, psi_q = known_flux(Id, Iq)
    n_t = 61                                            # samples over one electrical cycle
    z = {"time": np.linspace(0.0, 2.5, n_t), "time_unit": "ms", "gamma": gamma, "im": im,
         "psi_d": np.tile(psi_d, (n_t, 1)), "psi_q": np.tile(psi_q, (n_t, 1)), "torque": None}
    table, _, _ = build_winding_table(z, pole_pairs=4, conductor_area_m2=1.0)
    path = tmp_path / "known_input_data.csv"
    table.to_csv(path, index=False)
    return path
