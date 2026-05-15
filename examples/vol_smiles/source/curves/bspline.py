from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

from .base import CurveState

# Uses BSpline from scipy and their fitting functions.

@dataclass
class BSplineState(CurveState):
    """
    BSpline curve state.
    """
    knots: np.ndarray
    degree: int
    T: float
    coeffs: np.ndarray
    precision: float | np.ndarray = field(default=1.0, hash=False, compare=False)
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
        new = BSplineState(self.knots, self.degree, self.T, params.copy(), self.precision)
        new._dm_cache = self._dm_cache
        return new

    def copy(self) -> "BSplineState":
        return BSplineState(self.knots.copy(), self.degree, self.T, self.coeffs.copy(), self.precision)

    def with_T(self, new_T: float) -> "BSplineState":
        """Return a copy with updated T (used by roller when time-to-expiry decreases)."""
        new = BSplineState(self.knots, self.degree, new_T, self.coeffs.copy(), self.precision)
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

    def implied_vol(self, k: float | np.ndarray) -> float | np.ndarray:
        """Return implied vol directly from the spline."""
        k = np.asarray(k, dtype=float)
        scalar = k.ndim == 0
        k = np.atleast_1d(k)
        iv = self._basis(k) @ self.coeffs
        return float(np.maximum(iv[0], 1e-8)) if scalar else np.maximum(iv, 1e-8)

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """Return w(k) = σ(k)² · T, computed from the IV spline."""
        iv = self.implied_vol(k)
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
    Fit a cubic B-spline to (log-moneyness, implied-vol) data.

    Knots are clamped to the data range so the interior knots always fall
    strictly inside [k.min(), k.max()], satisfying scipy's Schoenberg-Whitney
    condition regardless of scipy version.
    """
    # Sort by k so make_lsq_spline receives monotone x.
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    if weights is not None:
        weights = weights[order]

    k_lo, k_hi = float(k[0]), float(k[-1])

    # Interior knots strictly inside the data range.
    interior = np.linspace(k_lo, k_hi, n_interior + 2)[1:-1]

    # Full clamped knot vector: endpoints repeated (degree+1) times.
    knots = np.concatenate([
        np.repeat(k_lo, degree + 1),
        interior,
        np.repeat(k_hi, degree + 1),
    ])
    n_coeffs = len(knots) - degree - 1

    try:
        spline = make_lsq_spline(k, iv, knots, k=degree, w=weights)
        coeffs = np.maximum(spline.c, 1e-8)
    except Exception:
        coeffs = np.full(n_coeffs, max(float(np.mean(iv)), 1e-8))

    return BSplineState(knots=knots, degree=degree, T=T, coeffs=coeffs)
