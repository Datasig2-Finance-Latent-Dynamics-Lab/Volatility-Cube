from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import CurveState


@dataclass
class SviRawState(CurveState):
    """SVI natural parameterization: (a, b, rho, m, sigma)."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_jw(self, T: float) -> "SviJWState":
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
        return SviRawState(a=params[0], b=params[1], rho=params[2], m=params[3], sigma=params[4])

    def copy(self) -> "SviRawState":
        return SviRawState(self.a, self.b, self.rho, self.m, self.sigma)

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

    T is a fixed hyperparameter required to convert to the
    natural parameterization. It is not included in parameters.
    Works as in BSplines.
    """
    v: float        # ATM variance rate  w_atm / T
    psi: float      # ATM skew
    p: float        # put wing slope  b * (1 - rho)
    c: float        # call wing slope b * (1 + rho)
    v_tilde: float  # minimum variance rate  w_min / T
    T: float        # time to expiry (not optimized)

    def to_raw(self) -> SviRawState:
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

        return SviRawState(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sigma))

    @property
    def n_params(self) -> int:
        return 5

    def parameters(self) -> np.ndarray:
        return np.array([self.v, self.psi, self.p, self.c, self.v_tilde])

    def from_parameters(self, params: np.ndarray) -> "SviJWState":
        return SviJWState(
            v=params[0], psi=params[1], p=params[2], c=params[3], v_tilde=params[4],
            T=self.T,
        )

    def copy(self) -> "SviJWState":
        return SviJWState(self.v, self.psi, self.p, self.c, self.v_tilde, self.T)

    def bounds(self) -> list[tuple[float | None, float | None]]:
        return [
            (1e-8, None),    # v
            (-0.5, 0.5),     # psi
            (1e-8, None),    # p 
            (1e-8, None),    # c
            (1e-8, None),    # v_tilde
        ]

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """
        Returns huge error if we get not reasonable SVI parameters.
        """
        try:
            return self.to_raw().total_variance(k)
        except ValueError:
            return np.full_like(np.asarray(k, dtype=float), 1e10)
