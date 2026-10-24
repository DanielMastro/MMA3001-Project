"""
models.py - flux models: predict (psi_d, psi_q) at any (Id, Iq).

Where this fits in the project
------------------------------
preprocess.py gives flux only at the FEA operating points (71 unique points for
Cobalt). The inverter needs flux, inductance and torque at ANY current, so a
model has to be fitted through those points. Four are compared (report
section 4, alternative solutions):

    constant_inductance()      baseline: psi_d = psi_pm + Ld*Id,  psi_q = Lq*Iq
    lookup_table()             baseline: straight lines between the FEA points,
                               what the Simulink lookup tables do now
    polynomial_regression(d)   candidate: least-squares fit of a polynomial of
                               degree d in Id and Iq (linear regression)
    neural_network(size)       candidate: small neural network, two hidden layers
                               of `size` neurons each

Every model has the same methods, so derive.py and evaluate.py can use any of them:

    model.fit(Id, Iq, psi_d, psi_q)    learn from FEA points (A, Wb)
    model.predict_flux(Id, Iq)         -> (psi_d, psi_q) in Wb
    model.flux_gradient(Id, Iq)        -> dict of dpsi/di in H (the incremental
                                          inductances)
    model.is_extrapolation(Id, Iq)     -> True where a query lies outside the
                                          FEA current range
    model.n_parameters                 numbers stored after fitting (memory)
    model.flops_per_eval()             rough arithmetic cost of one prediction

Per-unit scaling
----------------
Inside every model, currents are divided by the largest training current and
flux by the largest training flux, so the model works on numbers of order 1.
This keeps least squares well conditioned and lets the same settings
(polynomial degree, network size) work on a 113 A motor and a 620 A motor.
Inputs and outputs of the public methods are always in SI units.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def current_angle(Id, Iq):
    """Current angle beta in degrees, 07-07's definition: beta = atan2(-Id, Iq).
    0 = pure +q (motoring), 90 = pure -d (field weakening)."""
    return np.degrees(np.arctan2(-np.asarray(Id, dtype=float), np.asarray(Iq, dtype=float)))


# ============================================================================
# SHARED BEHAVIOUR (per-unit scaling, SI in/out, extrapolation flag)
# ============================================================================
class _FluxModel:
    """Common code for every flux model. Each model only supplies:
        _fit(x, y, pd, pq)   learn from per-unit currents x=Id/I_base, y=Iq/I_base
                             and per-unit flux pd=psi_d/psi_base, pq=psi_q/psi_base
        _predict(x, y)       -> (pd, pq) per-unit
        _gradient(x, y)      -> (n, 2, 2) array, [output, input] per-unit slopes;
                                optional (the default uses finite differences)
    """
    name = "flux model"

    def fit(self, Id, Iq, psi_d, psi_q):
        """Learn from FEA points. Arrays of equal length, SI units. Returns self."""
        Id, Iq, psi_d, psi_q = (np.asarray(v, dtype=float).ravel() for v in (Id, Iq, psi_d, psi_q))
        if not (len(Id) == len(Iq) == len(psi_d) == len(psi_q)):
            raise ValueError("Id, Iq, psi_d and psi_q must have the same length")
        if np.isnan(np.r_[Id, Iq, psi_d, psi_q]).any():
            raise ValueError("training data contains NaN")

        # Per-unit bases: largest training current and largest training flux
        self.I_base = float(np.max(np.hypot(Id, Iq)))
        self.psi_base = float(np.max(np.abs(np.r_[psi_d, psi_q])))

        # Training range, for the extrapolation flag: current angle span of the
        # non-zero points (the angle is undefined at zero current)
        on = np.hypot(Id, Iq) > 1e-6 * self.I_base
        beta = current_angle(Id[on], Iq[on])
        self.beta_range = (float(beta.min()), float(beta.max()))

        self._fit(Id / self.I_base, Iq / self.I_base, psi_d / self.psi_base, psi_q / self.psi_base)
        return self

    def predict_flux(self, Id, Iq):
        """Predicted (psi_d, psi_q) in Wb, same shape as the inputs."""
        Id, Iq = np.broadcast_arrays(np.asarray(Id, dtype=float), np.asarray(Iq, dtype=float))
        pd_pu, pq_pu = self._predict(Id.ravel() / self.I_base, Iq.ravel() / self.I_base)
        return (np.reshape(pd_pu * self.psi_base, Id.shape),
                np.reshape(pq_pu * self.psi_base, Id.shape))

    def flux_gradient(self, Id, Iq):
        """dpsi/di at (Id, Iq) in H, the incremental inductances. Keys:
        dpsid_did (Ld), dpsiq_diq (Lq), dpsid_diq and dpsiq_did (cross-coupling)."""
        Id, Iq = np.broadcast_arrays(np.asarray(Id, dtype=float), np.asarray(Iq, dtype=float))
        g = self._gradient(Id.ravel() / self.I_base, Iq.ravel() / self.I_base)
        k = self.psi_base / self.I_base          # per-unit slope -> henries
        shape = Id.shape
        return {"dpsid_did": np.reshape(g[:, 0, 0] * k, shape),
                "dpsid_diq": np.reshape(g[:, 0, 1] * k, shape),
                "dpsiq_did": np.reshape(g[:, 1, 0] * k, shape),
                "dpsiq_diq": np.reshape(g[:, 1, 1] * k, shape)}

    def _gradient(self, x, y, h=1e-4):
        """Default gradient: central differences on the model itself (used by
        lookup_table, whose slope is constant inside each triangle)."""
        g = np.empty((len(x), 2, 2))
        pd_p, pq_p = self._predict(x + h, y); pd_m, pq_m = self._predict(x - h, y)
        g[:, 0, 0] = (pd_p - pd_m) / (2 * h); g[:, 1, 0] = (pq_p - pq_m) / (2 * h)
        pd_p, pq_p = self._predict(x, y + h); pd_m, pq_m = self._predict(x, y - h)
        g[:, 0, 1] = (pd_p - pd_m) / (2 * h); g[:, 1, 1] = (pq_p - pq_m) / (2 * h)
        return g

    def is_extrapolation(self, Id, Iq, angle_tol_deg=0.5, current_tol=1e-3):
        """True where (Id, Iq) is outside the FEA range: a larger current
        magnitude than any training point, or a current angle outside the
        training angles. Predictions there are extrapolations.

        Small tolerances (0.5 deg, 0.1 % of current) stop points on the edge of
        the FEA sweep being flagged. The FEA's own angles sit about 0.04 deg off
        the axes because of the rotor-alignment step."""
        Id, Iq = np.broadcast_arrays(np.asarray(Id, dtype=float), np.asarray(Iq, dtype=float))
        I = np.hypot(Id, Iq)
        beta = current_angle(Id, Iq)
        lo, hi = self.beta_range
        inside = (I <= self.I_base * (1 + current_tol)) & (
            (I <= 1e-6 * self.I_base)
            | ((beta >= lo - angle_tol_deg) & (beta <= hi + angle_tol_deg)))
        return ~inside

    def __repr__(self):
        return f"{self.name}"


# ============================================================================
# BASELINE 1: CONSTANT INDUCTANCE
# ============================================================================
class _ConstantInductance(_FluxModel):
    """psi_d = psi_pm + Ld*Id, psi_q = Lq*Iq, with psi_pm, Ld, Lq found by least
    squares. What a single Ld/Lq pair assumes: no saturation, no cross-coupling."""
    name = "constant inductance"

    def _fit(self, x, y, pd, pq):
        A = np.column_stack([np.ones_like(x), x])                 # [1, Id] for psi_d
        (self.a, self.b), *_ = np.linalg.lstsq(A, pd, rcond=None)  # psi_d = a + b*Id
        self.c = np.linalg.lstsq(y[:, None], pq, rcond=None)[0][0] # psi_q = c*Iq

    def _predict(self, x, y):
        return self.a + self.b * x, self.c * y

    def _gradient(self, x, y):
        g = np.zeros((len(x), 2, 2))
        g[:, 0, 0] = self.b                                       # Ld (constant)
        g[:, 1, 1] = self.c                                       # Lq (constant)
        return g

    @property
    def n_parameters(self):
        return 3

    def flops_per_eval(self):
        return 4                                                   # two multiply-adds

    def fitted_values(self):
        """The fitted psi_pm [Wb], Ld [H] and Lq [H] in SI units."""
        k = self.psi_base / self.I_base
        return {"psi_pm": self.a * self.psi_base, "Ld": self.b * k, "Lq": self.c * k}


def constant_inductance():
    """Baseline flux model: psi_d = psi_pm + Ld*Id, psi_q = Lq*Iq (least squares)."""
    return _ConstantInductance()


# ============================================================================
# BASELINE 2: LOOKUP TABLE (piecewise-linear interpolation)
# ============================================================================
class _LookupTable(_FluxModel):
    """Straight lines between the FEA points. The (Id, Iq) plane is split into
    triangles with the FEA points as corners, and flux is interpolated linearly
    inside each triangle. It is exact at every FEA point. Outside the triangles
    it has no answer, so the nearest FEA point's value is used and the query is
    flagged as extrapolation."""
    name = "lookup table"

    def _fit(self, x, y, pd, pq):
        pts, vals = np.column_stack([x, y]), np.column_stack([pd, pq])
        self._linear = LinearNDInterpolator(pts, vals)            # inside the triangles
        self._nearest = NearestNDInterpolator(pts, vals)          # fallback outside
        self._n_points = len(x)

    def _predict(self, x, y):
        out = np.atleast_2d(self._linear(x, y))
        bad = np.isnan(out).any(axis=1)                           # outside every triangle
        if bad.any():
            out[bad] = self._nearest(x[bad], y[bad])
        return out[:, 0], out[:, 1]

    @property
    def n_parameters(self):
        return 4 * self._n_points                                  # Id, Iq, psi_d, psi_q per point

    def flops_per_eval(self):
        return 30                                                  # weights + blend, plus a triangle search


def lookup_table():
    """Baseline flux model: piecewise-linear interpolation between FEA points."""
    return _LookupTable()


# ============================================================================
# CANDIDATE 1: POLYNOMIAL REGRESSION
# ============================================================================
class _PolynomialRegression(_FluxModel):
    """psi = sum of c_ij * Id^i * Iq^j over all i + j <= degree, fitted by least
    squares. This is linear regression on polynomial terms. Degree 3 has 10
    terms per output (1, Id, Iq, Id^2, Id*Iq, Iq^2, Id^3, ...). The gradient is
    exact because a polynomial can be differentiated term by term."""

    def __init__(self, degree):
        self.degree = int(degree)
        self.name = f"polynomial regression (degree {self.degree})"
        # (i, j) exponent pairs for every term Id^i * Iq^j with i + j <= degree
        self.powers = [(i, t - i) for t in range(self.degree + 1) for i in range(t, -1, -1)]

    def _design(self, x, y):
        """Design matrix: one column per polynomial term."""
        return np.column_stack([x**i * y**j for i, j in self.powers])

    def _fit(self, x, y, pd, pq):
        A = self._design(x, y)
        if A.shape[0] < A.shape[1]:
            raise ValueError(f"degree {self.degree} needs at least {A.shape[1]} points")
        self.condition_number = float(np.linalg.cond(A))           # numerical stability check
        self.coef, *_ = np.linalg.lstsq(A, np.column_stack([pd, pq]), rcond=None)

    def _predict(self, x, y):
        out = self._design(x, y) @ self.coef
        return out[:, 0], out[:, 1]

    def _gradient(self, x, y):
        zero = np.zeros_like(x)
        # d/dx of x^i y^j = i x^(i-1) y^j  (and similarly for y); zero when i = 0
        dA_dx = np.column_stack([i * x**(i - 1) * y**j if i > 0 else zero for i, j in self.powers])
        dA_dy = np.column_stack([j * x**i * y**(j - 1) if j > 0 else zero for i, j in self.powers])
        g = np.empty((len(x), 2, 2))
        g[:, :, 0] = dA_dx @ self.coef                             # d(psi_d, psi_q)/dId
        g[:, :, 1] = dA_dy @ self.coef                             # d(psi_d, psi_q)/dIq
        return g

    @property
    def n_parameters(self):
        return self.coef.size                                      # 2 outputs x n_terms

    def flops_per_eval(self):
        n_terms = len(self.powers)
        return 5 * n_terms                                         # build each term + 2 multiply-adds


def polynomial_regression(degree):
    """Candidate flux model: least-squares polynomial of the given degree in
    Id and Iq. `degree` = highest total power (degree 3 includes Id^3, Id*Iq^2)."""
    return _PolynomialRegression(degree)


# ============================================================================
# CANDIDATE 2: NEURAL NETWORK
# ============================================================================
class _NeuralNetwork(_FluxModel):
    """A small fully connected neural network (scikit-learn's MLPRegressor,
    "multilayer perceptron"):

        (Id, Iq) -> [size neurons, tanh] -> [size neurons, tanh] -> (psi_d, psi_q)

    tanh is used rather than ReLU because it is smooth. A ReLU network is
    piecewise linear, so its derivative, the inductance, would come out in
    steps. Training uses L-BFGS, which suits small datasets. The gradient comes
    from the chain rule through the trained weights."""

    def __init__(self, size, seed=0):
        self.size = int(size)
        self.seed = seed
        self.name = f"neural network (size {self.size})"

    def _fit(self, x, y, pd, pq):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor

        X = np.column_stack([x, y])
        Y = np.column_stack([pd, pq])
        # Standardise the two outputs so both are weighted equally in training
        self._y_mean, self._y_std = Y.mean(axis=0), Y.std(axis=0)
        self._y_std[self._y_std == 0] = 1.0
        self._net = MLPRegressor(hidden_layer_sizes=(self.size, self.size), activation="tanh",
                                 solver="lbfgs", max_iter=5000, random_state=self.seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self._net.fit(X, (Y - self._y_mean) / self._y_std)
        self.converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    def _predict(self, x, y):
        out = self._net.predict(np.column_stack([x, y])) * self._y_std + self._y_mean
        return out[:, 0], out[:, 1]

    def _gradient(self, x, y):
        W1, W2, W3 = self._net.coefs_                 # (2, s), (s, s), (s, 2)
        b1, b2, _ = self._net.intercepts_
        X = np.column_stack([x, y])
        h1 = np.tanh(X @ W1 + b1)                      # hidden layer 1 outputs (n, s)
        h2 = np.tanh(h1 @ W2 + b2)                     # hidden layer 2 outputs (n, s)
        # Chain rule, layer by layer (d tanh(z)/dz = 1 - tanh(z)^2)
        dh1 = (1 - h1**2)[:, :, None] * W1.T[None, :, :]           # dh1/dX  (n, s, 2)
        dh2 = (1 - h2**2)[:, :, None] * np.einsum("km,nki->nmi", W2, dh1)   # dh2/dX
        dout = np.einsum("mo,nmi->noi", W3, dh2)                   # d(standardised out)/dX
        return dout * self._y_std[None, :, None]                   # undo the output scaling

    @property
    def n_parameters(self):
        return int(sum(w.size for w in self._net.coefs_) + sum(b.size for b in self._net.intercepts_))

    def flops_per_eval(self):
        s = self.size
        mult_adds = 2 * s + s * s + 2 * s                          # three weight layers
        return 2 * mult_adds + 2 * s * 20                          # + about 20 flops per tanh


def neural_network(size, seed=0):
    """Candidate flux model: neural network with two hidden layers.
    `size` = number of neurons in EACH of the two hidden layers (size 16 has
    354 weights and biases). `seed` fixes the random starting weights so
    results repeat."""
    return _NeuralNetwork(size, seed)
