"""
Self-contained parametric surface/smile representations and their fitters.

Ported from the dgraph state/curve classes (svi.py, ssvi.py, bspline.py) but stripped
of the Graph/State machinery — each class only needs to evaluate IV / total variance
and expose its coefficient vector (used by the regularised fitters).

Smile representations are 1-D in log-moneyness k (one per maturity):  SviRawState,
SviJWState, BSplineState.  Surface representations are 2-D in (k, T):  SSVISurfaceState.
All fits minimise weighted MSE in total-variance space (w = iv^2 * T).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline, make_lsq_spline


# ════════════════════════════════════════════════════════════════════════════
# SVI (raw + jump-wing) — per-maturity smile
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class SviRawState:
    """SVI natural parameterisation (a, b, rho, m, sigma) at fixed maturity T."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        w = self.a + self.b * (self.rho * (k - self.m)
                               + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))
        return np.maximum(w, 1e-10)

    def implied_vol(self, k):
        return np.sqrt(self.total_variance(k) / self.T)

    def to_jw(self, T: float | None = None) -> "SviJWState":
        T = T if T is not None else self.T
        hyp = np.sqrt(self.m ** 2 + self.sigma ** 2)
        w_atm = self.a + self.b * (-self.rho * self.m + hyp)
        return SviJWState(
            v=float(w_atm / T),
            psi=float(self.b * (self.rho - self.m / hyp)),
            p=float(self.b * (1.0 - self.rho)),
            c=float(self.b * (1.0 + self.rho)),
            v_tilde=float((self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho ** 2)) / T),
            T=T,
        )


@dataclass
class SviJWState:
    """SVI jump-wing parameterisation (v, psi, p, c, v_tilde) at fixed maturity T."""

    v: float
    psi: float
    p: float
    c: float
    v_tilde: float
    T: float

    def to_raw(self) -> SviRawState:
        w_t = self.v * self.T
        b = (self.c + self.p) / 2.0
        rho = (self.c - self.p) / (self.c + self.p)
        d = rho - self.psi / b
        if abs(d) >= 1.0:
            raise ValueError(f"JW inconsistent: |d|={abs(d):.4f} >= 1")
        sqrt_1md2 = np.sqrt(1.0 - d ** 2)
        sqrt_1mr2 = np.sqrt(1.0 - rho ** 2)
        denom = (1.0 - rho * d) / sqrt_1md2 - sqrt_1mr2
        if abs(denom) < 1e-12:
            raise ValueError("JW degenerate sigma")
        sigma = (w_t - self.v_tilde * self.T) / (b * denom)
        m = d * sigma / sqrt_1md2
        a = self.v_tilde * self.T - b * sigma * sqrt_1mr2
        return SviRawState(float(a), float(b), float(rho), float(m), float(sigma), self.T)

    def total_variance(self, k):
        if self.v_tilde > self.v:
            return np.full_like(np.asarray(k, dtype=float), 1e10)
        try:
            return self.to_raw().total_variance(k)
        except ValueError:
            return np.full_like(np.asarray(k, dtype=float), 1e10)

    def implied_vol(self, k):
        return np.sqrt(self.total_variance(k) / self.T)


def fit_svi(k, iv, T, weights=None) -> SviRawState:
    """Fit SVI raw parameterisation; weighted MSE in total-variance space."""
    k = np.asarray(k, float); iv = np.asarray(iv, float)
    w_obs = iv ** 2 * T

    def objective(params):
        a, b, rho, m, sigma = params
        w_fit = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        return float(np.average((w_fit - w_obs) ** 2, weights=weights))

    x0 = np.array([np.mean(w_obs) * 0.8, 0.1, -0.3, 0.0, 0.1])
    bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999), (-0.5, 0.5), (1e-4, None)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    a, b, rho, m, sigma = res.x
    return SviRawState(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T)


def fit_svi_jw(k, iv, T, weights=None) -> SviJWState:
    """Fit SVI jump-wing; falls back to fit_svi().to_jw() if JW invalid."""
    k = np.asarray(k, float); iv = np.asarray(iv, float)
    w_obs = iv ** 2 * T
    w_atm_guess = (float(np.interp(0.0, k, w_obs))
                   if k.min() <= 0 <= k.max() else float(np.mean(w_obs)))

    def objective(params):
        v, psi, p, c, v_tilde = params
        try:
            w_fit = SviJWState(v, psi, p, c, v_tilde, T).total_variance(k)
        except (ValueError, FloatingPointError):
            return 1e10
        return float(np.average((w_fit - w_obs) ** 2, weights=weights))

    v0 = w_atm_guess / T
    x0 = np.array([v0, -0.02, 0.08, 0.12, v0 * 0.8])
    bounds = [(1e-6, None), (-0.5, 0.5), (1e-6, None), (1e-6, None), (1e-6, None)]

    def obj_pen(params):
        v, _, _, _, v_tilde = params
        return objective(params) + 1e4 * max(v_tilde - v, 0.0) ** 2

    res = minimize(obj_pen, x0, method="L-BFGS-B", bounds=bounds)
    v, psi, p, c, v_tilde = res.x
    try:
        state = SviJWState(v, psi, p, c, v_tilde, T)
        state.to_raw()
        return state
    except (ValueError, FloatingPointError):
        return fit_svi(k, iv, T, weights).to_jw(T)


# ════════════════════════════════════════════════════════════════════════════
# B-spline — per-maturity smile (IV directly in coeffs; linear → closed-form regular.)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class BSplineState:
    """Cubic B-spline smile: coeffs are IV values; w = iv^2 * T."""

    knots: np.ndarray
    degree: int
    T: float
    coeffs: np.ndarray
    _dm_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def make_knots(cls, k_min=-0.5, k_max=0.5, n_interior=9, degree=3) -> np.ndarray:
        """Clamped knot vector with interior knots strictly inside [k_min, k_max]."""
        interior = np.linspace(k_min, k_max, n_interior + 2)[1:-1]
        return np.concatenate([
            np.full(degree + 1, k_min), interior, np.full(degree + 1, k_max)])

    @property
    def n_params(self) -> int:
        return len(self.coeffs)

    def design_matrix(self, k: np.ndarray) -> np.ndarray:
        """Dense (N, n_coeffs) B-spline design matrix at k (cached by content)."""
        k = np.asarray(k, dtype=float)
        key = k.tobytes()
        if key not in self._dm_cache:
            k_clipped = np.clip(k, self.knots[self.degree], self.knots[-self.degree - 1])
            self._dm_cache[key] = BSpline.design_matrix(
                k_clipped, self.knots, self.degree).toarray()
        return self._dm_cache[key]

    def implied_vol(self, k):
        k = np.asarray(k, dtype=float)
        scalar = k.ndim == 0
        k = np.atleast_1d(k)
        iv = self.design_matrix(k) @ self.coeffs
        return float(np.maximum(iv[0], 1e-8)) if scalar else np.maximum(iv, 1e-8)

    def total_variance(self, k):
        return self.implied_vol(k) ** 2 * self.T

    def with_coeffs(self, coeffs: np.ndarray, T: float | None = None) -> "BSplineState":
        new = BSplineState(self.knots, self.degree,
                           self.T if T is None else float(T), np.asarray(coeffs, float))
        new._dm_cache = self._dm_cache
        return new


def fit_bspline(k, iv, T, weights=None, n_interior=5, degree=3) -> BSplineState:
    """Least-squares cubic B-spline with data-driven clamped knots."""
    k = np.asarray(k, float); iv = np.asarray(iv, float)
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    if weights is not None:
        weights = np.asarray(weights)[order]

    k_lo, k_hi = float(k[0]), float(k[-1])
    interior = np.linspace(k_lo, k_hi, n_interior + 2)[1:-1]
    knots = np.concatenate([
        np.repeat(k_lo, degree + 1), interior, np.repeat(k_hi, degree + 1)])
    n_coeffs = len(knots) - degree - 1
    iv_max = float(iv.max())
    try:
        spline = make_lsq_spline(k, iv, knots, k=degree, w=weights)
        coeffs = spline.c
        # Reject ill-conditioned fits: a near-singular knot system produces
        # oscillating coefficients far outside the IV range (same guard as the
        # B-spline prior).  Fall back to a flat mean predictor.
        if _coeffs_blew_up(coeffs, iv_max):
            raise ValueError("ill-conditioned spline coefficients")
        coeffs = np.maximum(coeffs, 1e-8)
    except Exception:
        coeffs = np.full(n_coeffs, max(float(np.mean(iv)), 1e-8))
    return BSplineState(knots=knots, degree=degree, T=T, coeffs=coeffs)


_COEFF_MAX_MULT = 3.0       # reject a spline coeff above max(iv_max·mult, _COEFF_MAX_ABS)
_COEFF_MAX_ABS = 5.0
_COEFF_MIN = -0.1           # ...or below this (a near-singular knot system oscillates)


def _coeffs_blew_up(coeffs, iv_max) -> bool:
    """Ill-conditioned B-spline fit: non-finite or coefficients far outside the IV range.
    Shared guard for both `fit_bspline` and `fit_bspline_fixed`."""
    return (not np.all(np.isfinite(coeffs))
            or coeffs.max() > max(iv_max * _COEFF_MAX_MULT, _COEFF_MAX_ABS)
            or coeffs.min() < _COEFF_MIN)


def second_diff_matrix(p: int) -> np.ndarray:
    """Second-difference operator D2 (rows [1,-2,1]); ‖D2 c‖² is a P-spline curvature penalty."""
    if p < 3:
        return np.zeros((0, p))
    D = np.zeros((p - 2, p))
    for i in range(p - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return D


def fit_bspline_fixed(k, iv, T, knots, degree=3, weights=None, ridge=1e-3,
                      curvature=0.0) -> BSplineState:
    """Fit B-spline coeffs on a FIXED knot vector (so coeffs are comparable across days).

    Solves penalised weighted normal equations.  Two regularisers keep the fit sane where
    the data is sparse / doesn't span the knot range (else the unconstrained basis directions
    blow up):
      * ``ridge``     — a tiny shrink toward the flat mean-IV level (anchors the overall level);
      * ``curvature`` — a second-difference (P-spline) penalty λ‖D2 c‖² that smooths the
                        coefficient sequence, so empty regions interpolate flatly instead of
                        oscillating.  Negligible vs the data term once a node is well sampled,
                        so the fit still converges to the interpolant as observations grow.
    A blow-up guard (same as `fit_bspline`) falls back to a flat mean if the solve still
    produces coefficients far outside the IV range.
    """
    k = np.asarray(k, float); iv = np.asarray(iv, float)
    p = len(knots) - degree - 1
    state = BSplineState(knots=np.asarray(knots, float), degree=degree, T=T,
                         coeffs=np.zeros(p))
    if k.size == 0:
        state.coeffs = np.full(p, 0.2)
        return state
    B = state.design_matrix(k)                         # (N, p)
    w = np.ones_like(iv) if weights is None else np.asarray(weights, float)
    mean_iv = float(np.mean(iv))
    A = B.T @ (w[:, None] * B) + ridge * np.eye(p)
    if curvature > 0 and p >= 3:
        D2 = second_diff_matrix(p)
        A = A + curvature * (D2.T @ D2)
    b = B.T @ (w * iv) + ridge * mean_iv               # shrink toward flat mean
    coeffs = np.linalg.solve(A, b)
    iv_max = float(iv.max())
    if _coeffs_blew_up(coeffs, iv_max):
        coeffs = np.full(p, max(mean_iv, 1e-8))        # ill-conditioned → flat fallback
    state.coeffs = np.maximum(coeffs, 1e-8)
    return state


# ════════════════════════════════════════════════════════════════════════════
# SSVI — full surface in (k, T)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class SSVISurfaceState:
    """Power-law SSVI surface (Gatheral & Jacquier 2014)."""

    v_0: float
    v_inf: float
    kappa: float
    rho: float
    eta: float
    gamma: float

    def theta(self, T):
        T = np.asarray(T, dtype=float)
        kappa = max(self.kappa, 1e-10)
        return self.v_inf * T + (self.v_0 - self.v_inf) * (1.0 - np.exp(-kappa * T)) / kappa

    def phi(self, theta_T):
        theta_T = np.maximum(theta_T, 1e-12)
        return self.eta / (theta_T ** self.gamma * (1.0 + theta_T) ** (1.0 - self.gamma))

    def total_variance(self, k, T):
        k = np.asarray(k, dtype=float); T = np.asarray(T, dtype=float)
        theta_T = self.theta(T)
        phi_T = self.phi(theta_T)
        inner = phi_T * k + self.rho
        w = theta_T / 2.0 * (1.0 + self.rho * phi_T * k
                             + np.sqrt(inner ** 2 + (1.0 - self.rho ** 2)))
        return np.maximum(w, 1e-12)

    def implied_vol(self, k, T):
        return np.sqrt(self.total_variance(k, T) / np.maximum(np.asarray(T, float), 1e-12))


_SSVI_BOUNDS = [(1e-6, None), (1e-6, None), (1e-6, None), (-0.999, 0.999),
                (1e-6, 2.0), (1e-6, 0.5)]


def _ssvi_x0_guess(k, T, w_obs) -> np.ndarray:
    """Generic SSVI start: ATM-variance level from the data, default shape params."""
    atm_vars = []
    for t in np.unique(T):
        mask = T == t
        k_t, w_t = k[mask], w_obs[mask]
        atm_vars.append((t, w_t[int(np.argmin(np.abs(k_t)))]))
    if len(atm_vars) >= 2:
        t_arr = np.array([x[0] for x in atm_vars])
        w_arr = np.array([x[1] for x in atm_vars])
        v_guess = float(np.mean(w_arr / t_arr))
    else:
        v_guess = 0.04
    return np.array([v_guess, v_guess * 0.8, 1.0, -0.3, 0.5, 0.25])


def fit_ssvi(k, T, iv, weights=None) -> SSVISurfaceState:
    """Fit SSVI to all maturities jointly; weighted MSE in total-variance space."""
    k = np.asarray(k, float); T = np.asarray(T, float); iv = np.asarray(iv, float)
    w_obs = iv ** 2 * T
    x0 = _ssvi_x0_guess(k, T, w_obs)

    def na_penalty(p):
        eta, rho = p[4], p[3]
        return 1e4 * max(eta * (1.0 + abs(rho)) - 2.0, 0.0) ** 2

    def objective(p):
        state = SSVISurfaceState(*p)
        try:
            w_fit = state.total_variance(k, T)
        except Exception:
            return 1e10
        return float(np.average((w_fit - w_obs) ** 2, weights=weights)) + na_penalty(p)

    res = minimize(objective, x0, method="L-BFGS-B", bounds=_SSVI_BOUNDS)
    return SSVISurfaceState(*res.x)


# ── fast, warm-startable SSVI with an analytic gradient ───────────────────────
def ssvi_tv_grad(p, k, T):
    """SSVI total variance and its analytic Jacobian w.r.t. p=(v0,v_inf,kappa,rho,eta,gamma).

    Returns ``(w, dw)`` with ``w`` shape (n,) and ``dw`` shape (6, n) — the per-point
    ∂w/∂p.  Shared by the fast fitter (objective gradient) and the Kalman-SSVI model
    (observation Jacobian for the EKF covariance update / iv_std).
    """
    v0, v_inf, kappa, rho, eta, gamma = p
    kappa = max(float(kappa), 1e-10)
    k = np.asarray(k, float); T = np.asarray(T, float)
    e = np.exp(-kappa * T)
    g = (1.0 - e) / kappa
    theta = np.maximum(v0 * g + v_inf * (T - g), 1e-12)
    phi = eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))
    inner = phi * k + rho
    S = np.sqrt(np.maximum(inner ** 2 + (1.0 - rho ** 2), 1e-300))
    F = 1.0 + rho * phi * k + S
    w = np.maximum(0.5 * theta * F, 1e-12)

    dw_dphi = 0.5 * theta * (rho * k + inner * k / S)
    dphi_dtheta = phi * (-gamma / theta - (1.0 - gamma) / (1.0 + theta))
    dw_dtheta = 0.5 * F + dw_dphi * dphi_dtheta            # explicit + through phi
    dg_dkappa = (kappa * T * e - 1.0 + e) / (kappa ** 2)

    dw = np.empty((6, k.size))
    dw[0] = dw_dtheta * g                                  # v0
    dw[1] = dw_dtheta * (T - g)                            # v_inf
    dw[2] = dw_dtheta * (v0 - v_inf) * dg_dkappa           # kappa
    dw[3] = 0.5 * theta * phi * k * (1.0 + 1.0 / S)        # rho (explicit only)
    dw[4] = dw_dphi * (phi / eta)                          # eta
    dw[5] = dw_dphi * (phi * np.log((1.0 + theta) / theta))  # gamma
    return w, dw


def _ssvi_obj_grad(p, k, T, w_obs, weights=None):
    """Weighted total-variance MSE + no-arb penalty, with analytic gradient."""
    w, dw = ssvi_tv_grad(p, k, T)
    resid = w - w_obs
    wt = np.ones_like(resid) if weights is None else np.asarray(weights, float)
    W = wt.sum()
    obj = float(np.sum(wt * resid ** 2) / W)
    grad = (2.0 / W) * (dw * (wt * resid)).sum(axis=1)
    eta, rho = p[4], p[3]
    u = eta * (1.0 + abs(rho)) - 2.0                       # no-arb: eta(1+|rho|) <= 2
    if u > 0:
        obj += 1e4 * u ** 2
        grad[4] += 1e4 * 2.0 * u * (1.0 + abs(rho))
        grad[3] += 1e4 * 2.0 * u * eta * np.sign(rho)
    return obj, grad


def fit_ssvi_fast(k, T, iv, x0=None, weights=None, maxiter=80) -> SSVISurfaceState:
    """Warm-startable SSVI fit with an analytic gradient — for bulk per-(asset, day) fits.

    Same model and objective as :func:`fit_ssvi`, but (a) seeds from ``x0`` (e.g.
    yesterday's params) so consecutive surfaces converge in a handful of iterations, and
    (b) supplies ``jac`` so each L-BFGS-B step costs one objective eval, not seven.  Use
    this for the training-time series over every (asset, day); keep :func:`fit_ssvi` for
    one-off per-day fits (the temporal/graph models).
    """
    k = np.asarray(k, float); T = np.asarray(T, float); iv = np.asarray(iv, float)
    w_obs = iv ** 2 * T
    if x0 is None:
        x0 = _ssvi_x0_guess(k, T, w_obs)
    res = minimize(_ssvi_obj_grad, np.asarray(x0, float), args=(k, T, w_obs, weights),
                   method="L-BFGS-B", jac=True, bounds=_SSVI_BOUNDS,
                   options={"maxiter": maxiter})
    return SSVISurfaceState(*res.x)
