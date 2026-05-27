from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

from .base import SurfaceState


@dataclass
class SSVISurfaceState(SurfaceState):
    """
    Power-law SSVI surface (Gatheral & Jacquier 2014).

    w(k, T) = θ(T)/2 · [1 + ρ·φ(T)·k + sqrt((φ(T)·k + ρ)² + (1 − ρ²))]

    ATM total-variance term structure:
        θ(T) = v_inf·T + (v_0 − v_inf)·(1 − exp(−κ·T)) / κ

    Wing function (power-law):
        φ(θ) = η / (θ^γ · (1 + θ)^(1−γ))

    Parameters: (v_0, v_inf, kappa, rho, eta, gamma)

    Sufficient no-static-arbitrage conditions (Gatheral-Jacquier):
        0 < γ ≤ 1/2
        η · (1 + |ρ|) ≤ 2
        θ(T) strictly increasing  ← satisfied when v_0, v_inf, κ > 0
    """

    v_0: float      # short-term ATM variance rate  (dθ/dT at T→0)
    v_inf: float    # long-term ATM variance rate   (dθ/dT as T→∞)
    kappa: float    # mean-reversion speed of term structure
    rho: float      # correlation / overall skew
    eta: float      # wing-scaling level
    gamma: float    # power-law exponent
    precision: float | np.ndarray = field(default=1.0, hash=False, compare=False)

    # ------------------------------------------------------------------
    # Core surface formulae
    # ------------------------------------------------------------------

    def theta(self, T: float | np.ndarray) -> np.ndarray:
        """ATM total variance at maturity T."""
        T = np.asarray(T, dtype=float)
        kappa = max(self.kappa, 1e-10)
        return self.v_inf * T + (self.v_0 - self.v_inf) * (1.0 - np.exp(-kappa * T)) / kappa

    def phi(self, theta_T: np.ndarray) -> np.ndarray:
        """Power-law wing function evaluated at θ(T)."""
        theta_T = np.maximum(theta_T, 1e-12)
        return self.eta / (theta_T ** self.gamma * (1.0 + theta_T) ** (1.0 - self.gamma))

    def total_variance(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        T = np.asarray(T, dtype=float)
        theta_T = self.theta(T)
        phi_T   = self.phi(theta_T)
        inner   = phi_T * k + self.rho
        w = theta_T / 2.0 * (1.0 + self.rho * phi_T * k + np.sqrt(inner ** 2 + (1.0 - self.rho ** 2)))
        return np.maximum(w, 1e-12)

    # ------------------------------------------------------------------
    # State interface
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return 6

    def parameters(self) -> np.ndarray:
        return np.array([self.v_0, self.v_inf, self.kappa, self.rho, self.eta, self.gamma])

    def from_parameters(self, params: np.ndarray) -> "SSVISurfaceState":
        return SSVISurfaceState(
            v_0=params[0], v_inf=params[1], kappa=params[2],
            rho=params[3], eta=params[4], gamma=params[5],
            precision=self.precision,
        )

    def copy(self) -> "SSVISurfaceState":
        return SSVISurfaceState(
            self.v_0, self.v_inf, self.kappa,
            self.rho, self.eta, self.gamma, self.precision,
        )

    def bounds(self) -> list[tuple[float | None, float | None]]:
        return [
            (1e-6, None),    # v_0
            (1e-6, None),    # v_inf
            (1e-6, None),    # kappa
            (-0.999, 0.999), # rho
            (1e-6, 2.0),     # eta  (eta*(1+|rho|) <= 2 enforced softly)
            (1e-6, 0.5),     # gamma
        ]


def fit_ssvi(
    k: np.ndarray,
    T: np.ndarray,
    iv: np.ndarray,
    weights: np.ndarray | None = None,
) -> SSVISurfaceState:
    """
    Fit SSVISurfaceState to a collection of (k, T, iv) observations from all maturities
    simultaneously.  Minimises weighted MSE in total-variance space.
    """
    w_obs = iv ** 2 * T

    # Initial guess: fit ATM variance rate from data
    T_unique = np.unique(T)
    atm_vars = []
    for t in T_unique:
        mask = T == t
        k_t, w_t = k[mask], w_obs[mask]
        # ATM total variance ≈ value nearest k=0
        idx = np.argmin(np.abs(k_t))
        atm_vars.append((t, w_t[idx]))

    if len(atm_vars) >= 2:
        t_arr = np.array([x[0] for x in atm_vars])
        w_arr = np.array([x[1] for x in atm_vars])
        v_guess = float(np.mean(w_arr / t_arr))
    else:
        v_guess = 0.04

    x0 = np.array([v_guess, v_guess * 0.8, 1.0, -0.3, 0.5, 0.25])

    bounds = [
        (1e-6, None),    # v_0
        (1e-6, None),    # v_inf
        (1e-6, None),    # kappa
        (-0.999, 0.999), # rho
        (1e-6, 2.0),     # eta
        (1e-6, 0.5),     # gamma
    ]

    def na_penalty(params: np.ndarray) -> float:
        """Soft penalty for eta*(1+|rho|) > 2."""
        eta, rho = params[4], params[3]
        return 1e4 * max(eta * (1.0 + abs(rho)) - 2.0, 0.0) ** 2

    def objective(params: np.ndarray) -> float:
        v_0, v_inf, kappa, rho, eta, gamma = params
        state = SSVISurfaceState(v_0, v_inf, kappa, rho, eta, gamma)
        try:
            w_fit = state.total_variance(k, T)
        except Exception:
            return 1e10
        resid = w_fit - w_obs
        return float(np.average(resid ** 2, weights=weights)) + na_penalty(params)

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    v_0, v_inf, kappa, rho, eta, gamma = result.x
    return SSVISurfaceState(v_0=v_0, v_inf=v_inf, kappa=kappa, rho=rho, eta=eta, gamma=gamma)
