"""
Fixed tensor-product B-spline basis on the shared (lm, T) grid.

A drop-in sibling of `FactorBasis` (same `observation_matrix` / `mean_at` / `transform`
interface) so `KalmanFactorModel` can run a Kalman filter on a surface's *B-spline
coefficients* exactly as it does on PCA factor scores — the only difference is the linear
map from coefficients to IV.

Where `FactorBasis` learns its modes from data (PCA), this basis is *fixed*: a clamped
cubic tensor-product B-spline over (log-moneyness, maturity), one independent coefficient
block per asset.  So the coefficient vector is laid out as per-asset blocks
  c = [ c_asset0 , c_asset1 , … ],   |c_a| = n_lm_basis * n_T_basis,
which is what lets the Kalman transition's cross-asset coupling be zeroed block-wise for
the no-cross-asset ablation (unlike PCA, whose modes mix assets).

The surface is modelled directly (mean ≡ 0): iv ≈ B(lm, T) c, with B the tensor design
matrix.  `transform` fits the coefficients of a day's gridded surface by least squares.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from surfacelab.models.factors import FixedGrid


def _clamped_knots(lo: float, hi: float, n_basis: int, degree: int) -> np.ndarray:
    """Clamped-uniform knot vector giving exactly `n_basis` basis functions of `degree`."""
    n_interior = max(n_basis - degree - 1, 0)
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1] if n_interior else np.array([])
    return np.concatenate([np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])


def _design_1d(x, knots, degree) -> np.ndarray:
    """Dense 1-D B-spline design matrix at x (clamped to the knot domain so wing/long-T
    points outside the fitted range evaluate against the boundary basis instead of raising)."""
    lo, hi = knots[degree], knots[-degree - 1]
    xc = np.clip(np.asarray(x, float), lo, hi)
    return np.asarray(BSpline.design_matrix(xc, knots, degree).todense())


class BSplineBasis:
    """Per-asset fixed tensor-product cubic B-spline basis with the FactorBasis interface."""

    def __init__(self, grid: FixedGrid, n_lm: int = 6, n_T: int = 4, degree: int = 3):
        self.grid = grid
        self.degree = degree
        # knot domains span the grid's own (lm, T) extent
        self.lm_knots = _clamped_knots(grid.lm_vals.min(), grid.lm_vals.max(), n_lm, degree)
        self.T_knots = _clamped_knots(grid.T_vals.min(), grid.T_vals.max(), n_T, degree)
        self.n_lm = n_lm
        self.n_T = n_T
        self.coeffs_per_asset = n_lm * n_T
        self.k = self.coeffs_per_asset * grid.n_assets        # total state dim (matches FactorBasis.k)
        self.mean_ = np.zeros(grid.n_total)                   # surface modelled directly (no mean)
        # grid design (same for every asset) + its pseudo-inverse for least-squares coeff fits
        self._Bgrid = self._tensor_design(grid.lm_grid, grid.T_grid)   # (n_grid, coeffs_per_asset)
        self._Bpinv = np.linalg.pinv(self._Bgrid)
        self._col_mean: np.ndarray | None = None

    # ── design matrices ─────────────────────────────────────────────────────────
    def _tensor_design(self, lm, T) -> np.ndarray:
        """Row-wise Kronecker of the 1-D designs → (N, coeffs_per_asset)."""
        Blm = _design_1d(lm, self.lm_knots, self.degree)      # (N, n_lm)
        Bt = _design_1d(T, self.T_knots, self.degree)         # (N, n_T)
        return np.einsum("ij,ik->ijk", Blm, Bt).reshape(len(Blm), -1)

    def observation_matrix(self, feats, aids) -> np.ndarray:
        """H: (N, k) — each point's tensor-basis row placed in its asset's coefficient block."""
        feats = np.asarray(feats, float)
        aids = np.asarray(aids)
        H = np.zeros((len(feats), self.k), dtype=np.float64)
        cpa = self.coeffs_per_asset
        for a in range(self.grid.n_assets):
            m = aids == a
            if m.any():
                H[m, a * cpa:(a + 1) * cpa] = self._tensor_design(feats[m, 0], feats[m, 1])
        return H

    def mean_at(self, feats, aids) -> np.ndarray:
        return np.zeros(len(np.asarray(feats)), dtype=np.float64)

    # ── coefficient fit ─────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray) -> "BSplineBasis":
        """Record a per-coefficient mean (over days) as the fill value for days an asset is
        missing; the basis itself is fixed, so there is nothing else to learn."""
        Z = self.transform(X, _fill=False)
        self._col_mean = np.where(np.all(np.isnan(Z), axis=0), 0.0, np.nanmean(Z, axis=0))
        return self

    def transform(self, X: np.ndarray, _fill: bool = True) -> np.ndarray:
        """Least-squares B-spline coefficients per (day, asset) from gridded surfaces X:
        (n_days, n_total). Days where an asset is absent (all-NaN grid block) yield NaN, then
        filled with the per-coefficient training mean so the AR fit sees no holes."""
        X = np.atleast_2d(X)
        n_days = X.shape[0]
        cpa, ng = self.coeffs_per_asset, self.grid.n_grid
        Z = np.full((n_days, self.k), np.nan)
        for a in range(self.grid.n_assets):
            block = X[:, a * ng:(a + 1) * ng]                 # (n_days, n_grid)
            ok = ~np.isnan(block).any(axis=1)
            if ok.any():
                Z[ok, a * cpa:(a + 1) * cpa] = block[ok] @ self._Bpinv.T
        if _fill and self._col_mean is not None:
            bad = np.isnan(Z)
            if bad.any():
                Z[bad] = np.take(self._col_mean, np.where(bad)[1])
        return Z
