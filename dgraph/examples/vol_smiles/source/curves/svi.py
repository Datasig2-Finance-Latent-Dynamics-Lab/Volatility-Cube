from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

from .base import CurveState


@dataclass
class SviRawState(CurveState):
    """SVI natural parameterization: (a, b, rho, m, sigma). T and precision are not optimized."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    precision: float | np.ndarray = field(default=1.0, hash=False, compare=False)

    def with_T(self, new_T: float) -> "SviRawState":
        """
        Returns the same state with a different time to expiry.
        """
        return SviRawState(self.a, self.b, self.rho, self.m, self.sigma, new_T, self.precision)

    def to_jw(self, T: float | None = None) -> "SviJWState":
        """
        Turns the state into one parametrized by JW parameters.
        """
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

    @property
    def n_params(self) -> int:
        return 5

    def parameters(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    def from_parameters(self, params: np.ndarray) -> "SviRawState":
        """
        Creates a raw svi state with equalt time to expiry and precision but different parameters.
        """
        return SviRawState(
            a=params[0], b=params[1], rho=params[2], m=params[3], sigma=params[4],
            T=self.T, precision=self.precision,
        )

    def copy(self) -> "SviRawState":
        return SviRawState(self.a, self.b, self.rho, self.m, self.sigma, self.T, self.precision)

    def bounds(self) -> list[tuple[float | None, float | None]]:
        return [
            (1e-8, None),    # a
            (1e-8, None),    # b
            (-0.999, 0.999), # rho
            (-1.0, 1.0),     # m
            (1e-8, None),    # sigma
        ]

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        k = np.asarray(k)
        w = self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )
        return np.maximum(w, 1e-10)


@dataclass
class SviJWState(CurveState):
    """
    SVI Jump-Wing parameterization: (v, psi, p, c, v_tilde).
    T and precision are not optimized.
    """
    v: float        # ATM variance rate  w_atm / T
    psi: float      # ATM skew
    p: float        # put wing slope  b * (1 - rho)
    c: float        # call wing slope b * (1 + rho)
    v_tilde: float  # minimum variance rate  w_min / T
    T: float        # time to expiry (not optimized)
    precision: float | np.ndarray = field(default=1.0, hash=False, compare=False)

    def with_T(self, new_T: float) -> "SviJWState":
        return SviJWState(self.v, self.psi, self.p, self.c, self.v_tilde, new_T, self.precision)

    def to_raw(self) -> SviRawState:
        """
        Returns the same state parametrized by raw svi.
        """
        w_t = self.v * self.T
        b = (self.c + self.p) / 2.0
        rho = (self.c - self.p) / (self.c + self.p)
        d = rho - self.psi / b

        if abs(d) >= 1.0:
            raise ValueError(
                f"JW parameters are inconsistent: |d|={abs(d):.4f} >= 1. "
                "Check that psi is not too large relative to p+c."
            )

        sqrt_1md2 = np.sqrt(1.0 - d ** 2)
        sqrt_1mr2 = np.sqrt(1.0 - rho ** 2)

        denom = (1.0 - rho * d) / sqrt_1md2 - sqrt_1mr2
        if abs(denom) < 1e-12:
            raise ValueError("JW parameters lead to a degenerate sigma (minimum at ATM).")

        sigma = (w_t - self.v_tilde * self.T) / (b * denom)

        m = d * sigma / sqrt_1md2
        a = self.v_tilde * self.T - b * sigma * sqrt_1mr2

        return SviRawState(
            a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sigma),
            T=self.T, precision=self.precision,
        )

    @property
    def n_params(self) -> int:
        return 5

    def parameters(self) -> np.ndarray:
        return np.array([self.v, self.psi, self.p, self.c, self.v_tilde])

    def from_parameters(self, params: np.ndarray) -> "SviJWState":
        return SviJWState(
            v=params[0], psi=params[1], p=params[2], c=params[3], v_tilde=params[4],
            T=self.T, precision=self.precision,
        )

    def copy(self) -> "SviJWState":
        return SviJWState(self.v, self.psi, self.p, self.c, self.v_tilde, self.T, self.precision)

    def bounds(self) -> list[tuple[float | None, float | None]]:
        return [
            (1e-8, None),    # v
            (-0.5, 0.5),     # psi
            (1e-8, None),    # p
            (1e-8, None),    # c
            (1e-8, None),    # v_tilde
        ]

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        if self.v_tilde > self.v:
            # Minimum variance exceeds ATM variance — physically invalid.
            return np.full_like(np.asarray(k, dtype=float), 1e10)
        try:
            return self.to_raw().total_variance(k)
        except ValueError:
            return np.full_like(np.asarray(k, dtype=float), 1e10)


def fit_svi(k: np.ndarray, iv: np.ndarray, T: float, weights: np.ndarray | None = None) -> SviRawState:
    """Fit SVI raw parameterisation to (log-moneyness, implied-vol) data. Minimises weighted MSE in total-variance space."""
    w_obs = iv ** 2 * T

    def objective(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        w_fit = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        return float(np.average((w_fit - w_obs) ** 2, weights=weights))

    x0 = np.array([np.mean(w_obs) * 0.8, 0.1, -0.3, 0.0, 0.1])
    bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999), (-0.5, 0.5), (1e-4, None)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    a, b, rho, m, sigma = result.x
    return SviRawState(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T)


def fit_svi_jw(k: np.ndarray, iv: np.ndarray, T: float, weights: np.ndarray | None = None) -> SviJWState:
    """
    Fit SVI Jump-Wing parameterisation to (log-moneyness, implied-vol) data.
    Falls back to fit_svi().to_jw() if the JW fit yields invalid parameters.
    """
    w_obs = iv ** 2 * T
    w_atm_guess = float(np.interp(0.0, k, w_obs)) if k.min() <= 0 <= k.max() else float(np.mean(w_obs))

    def objective(jw_params: np.ndarray) -> float:
        v, psi, p, c, v_tilde = jw_params
        try:
            w_fit = SviJWState(v, psi, p, c, v_tilde, T).total_variance(k)
        except (ValueError, FloatingPointError):
            return 1e10
        return float(np.average((w_fit - w_obs) ** 2, weights=weights))

    v0 = w_atm_guess / T
    x0 = np.array([v0, -0.02, 0.08, 0.12, v0 * 0.8])
    bounds = [(1e-6, None), (-0.5, 0.5), (1e-6, None), (1e-6, None), (1e-6, None)]

    def objective_with_penalty(jw_params: np.ndarray) -> float:
        v, _, _, _, v_tilde = jw_params
        return objective(jw_params) + 1e4 * max(v_tilde - v, 0.0) ** 2

    result = minimize(objective_with_penalty, x0, method="L-BFGS-B", bounds=bounds)
    v, psi, p, c, v_tilde = result.x

    try:
        state = SviJWState(v, psi, p, c, v_tilde, T)
        state.to_raw()
        return state
    except (ValueError, FloatingPointError):
        return fit_svi(k, iv, T, weights).to_jw(T)
