from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

from .base import CurveState

# Uses BSPline from scipy and their fitting functions.

@dataclass
class BSplineState(CurveState):
    """
    BSpline curve state.
    """
    knots: np.ndarray
    degree: int
    T: float
    coeffs: np.ndarray
    _dm_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def make_knots(
        cls,
        k_min: float = -1.5,
        k_max: float = 1.5,
        n_interior: int = 5,
        degree: int = 3,
    ) -> np.ndarray:
        """
        Makes the knots for the BSpline.
        """
        interior = np.linspace(k_min, k_max, n_interior + 2)[1:-1]
        return np.concatenate([
            np.full(degree + 1, k_min), # ""
            interior,
            np.full(degree + 1, k_max), # These ensure max smoothness at endpoints.
        ])

    @property
    def n_params(self) -> int:
        return len(self.coeffs)

    def parameters(self) -> np.ndarray:
        return self.coeffs.copy()

    def from_parameters(self, params: np.ndarray) -> "BSplineState":
        # Share the cache — knots/degree are identical so the basis rows are reusable.
        new = BSplineState(self.knots, self.degree, self.T, params.copy())
        new._dm_cache = self._dm_cache
        return new

    def copy(self) -> "BSplineState":
        return BSplineState(self.knots.copy(), self.degree, self.T, self.coeffs.copy())

    def with_T(self, new_T: float) -> "BSplineState":
        """Return a copy with updated T (used by roller when time-to-expiry decreases)."""
        new = BSplineState(self.knots, self.degree, new_T, self.coeffs.copy())
        new._dm_cache = self._dm_cache
        return new

    def bounds(self) -> list[tuple[float | None, float | None]]:
        """Coefficient bounds are fairly general."""
        return [(1e-8, None)] * self.n_params

    def _basis(self, k: np.ndarray) -> np.ndarray:
        """Return precomputed dense design matrix for this k-array (cached by content)."""
        key = k.tobytes()
        if key not in self._dm_cache:
            k_clipped = np.clip(k, self.knots[self.degree], self.knots[-self.degree - 1])
            self._dm_cache[key] = BSpline.design_matrix(
                k_clipped, self.knots, self.degree
            ).toarray()
        return self._dm_cache[key]

    def implied_vol(self, k: float | np.ndarray, T: float) -> float | np.ndarray:
        """Return implied vol directly from the spline (T argument is not used)."""
        k = np.asarray(k, dtype=float)
        scalar = k.ndim == 0
        k = np.atleast_1d(k)
        iv = self._basis(k) @ self.coeffs
        return float(np.maximum(iv[0], 1e-8)) if scalar else np.maximum(iv, 1e-8)

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """Return w(k) = σ(k)² · T, computed from the IV spline."""
        iv = self.implied_vol(k, self.T)
        return iv ** 2 * self.T


def fit_bspline(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
    n_interior: int = 5,
    degree: int = 3,
) -> BSplineState:
    """

    """
    k_min = min(k.min() - 0.05, -1.5)
    k_max = max(k.max() + 0.05,  1.5)
    knots = BSplineState.make_knots(k_min, k_max, n_interior, degree)

    n_coeffs = len(knots) - degree - 1

    try:
        spline = make_lsq_spline(k, iv, knots[degree:-degree], k=degree, w=weights)
        coeffs = spline.c
    except Exception:
        coeffs = np.full(n_coeffs, np.mean(iv))

    # Ensure enough coefficients (make_lsq_spline may return fewer)
    if len(coeffs) < n_coeffs:
        coeffs = np.concatenate([coeffs, np.full(n_coeffs - len(coeffs), coeffs[-1])])

    coeffs = np.maximum(coeffs, 1e-8)
    return BSplineState(knots=knots, degree=degree, T=T, coeffs=coeffs)
