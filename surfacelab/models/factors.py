"""
Joint multi-asset functional-PCA basis on a fixed (lm, T) grid.

Each day's surface is interpolated onto a per-asset fixed grid and stacked into one
vector x = [iv_asset0_grid, iv_asset1_grid, ...].  PCA across days yields a basis
  x ≈ mean + B z,   z ∈ R^k (factors).
`observation_matrix` / `mean_at` evaluate the basis at arbitrary (lm, T) query points
so factors can be fit from (or a Kalman update run on) sparse daily observations.

Ported from `KalmanFilter/grid.py` (FixedGrid) and `KalmanFilter/pca.py` (SurfacePCA),
merged here so both the PCA surface model and the Kalman factor model share one basis.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.decomposition import PCA


def interp_linear_nearest(pts, vals, query) -> np.ndarray:
    """Linear scattered-data interpolation with nearest-neighbour fallback.

    Robust to the degenerate inputs a sparse context produces: with <3 points, or points
    that are collinear / co-located (so Qhull can't build a 2-D triangulation and `griddata`
    raises), it falls back to nearest-neighbour everywhere; nearest also fills any query
    point outside the convex hull.  Never raises as long as there is ≥1 point."""
    pts = np.asarray(pts, dtype=float)
    vals = np.asarray(vals, dtype=float)
    query = np.atleast_2d(np.asarray(query, dtype=float))
    if len(pts) == 0:
        return np.full(len(query), np.nan)
    if len(pts) < 3:
        return griddata(pts, vals, query, method="nearest")
    try:
        out = griddata(pts, vals, query, method="linear")
    except Exception:
        out = np.full(len(query), np.nan)          # degenerate hull (collinear points)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = griddata(pts, vals, query[bad], method="nearest")
    return out


class FixedGrid:
    """Fixed (lm, T) grid per asset; flat order index = asset*n_grid + j."""

    def __init__(self, lm_vals, T_vals, n_assets: int):
        self.lm_vals = np.asarray(lm_vals, dtype=np.float64)
        self.T_vals = np.asarray(T_vals, dtype=np.float64)
        self.n_assets = n_assets
        lm, T = np.meshgrid(self.lm_vals, self.T_vals, indexing="ij")
        self.lm_grid = lm.ravel()
        self.T_grid = T.ravel()
        self.grid_pts = np.stack([self.lm_grid, self.T_grid], axis=1)
        self.n_grid = len(self.lm_grid)
        self.n_total = self.n_grid * n_assets

    def interp_day(self, feats, aids, ivs) -> np.ndarray:
        """Interpolate one day's scatter onto the grid → (n_assets, n_grid), NaN if <3 pts."""
        out = np.full((self.n_assets, self.n_grid), np.nan, dtype=np.float64)
        for a in range(self.n_assets):
            mask = aids == a
            if mask.sum() < 3:
                continue
            out[a] = interp_linear_nearest(feats[mask].astype(np.float64),
                                           ivs[mask].astype(np.float64), self.grid_pts)
        return out

    def stack_days(self, dataset, indices) -> np.ndarray:
        """Interpolate each day onto the grid and stack → (len(indices), n_total).

        Shared by the PCA and Kalman factor models, which both PCA the same day-stacked
        grid matrix."""
        X = np.full((len(indices), self.n_total), np.nan)
        for i, t in enumerate(indices):
            m = dataset.valid_mask(t)
            X[i] = self.interp_day(dataset.query_feats[t, m], dataset.asset_ids[t, m],
                                   dataset.targets[t, m]).ravel()
        return X


class FactorBasis:
    """Joint PCA of the multi-asset surface on a FixedGrid + arbitrary-point evaluation."""

    def __init__(self, n_components: int, grid: FixedGrid):
        self.k = n_components
        self.grid = grid
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.evr_: np.ndarray | None = None
        self._comp_interps = None
        self._mean_interps = None

    # ── fitting ───────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, weights: np.ndarray | None = None) -> "FactorBasis":
        """X: (n_days, n_total) grid surfaces (NaN allowed → filled with column mean).

        weights: optional per-grid-column weight (length n_total). The modes then minimise a
        WEIGHTED reconstruction error Σ_g w_g (x_g − x̂_g)², so up-weighting one asset's columns
        spends the (few) modes on reconstructing THAT asset well, at the cost of the others.
        Implemented by scaling columns by √w before the SVD and unscaling the modes back to
        IV space — so observation_matrix/mean_at are unchanged. weights=None → ordinary PCA."""
        col_mean = np.nanmean(X, axis=0)
        Xf = np.where(np.isnan(X), col_mean, X)
        self.mean_ = col_mean
        w = np.ones(X.shape[1]) if weights is None else np.asarray(weights, float)
        self.w_col = w
        sw = np.sqrt(w)
        pca = PCA(n_components=min(self.k, Xf.shape[0], Xf.shape[1]))
        pca.fit((Xf - self.mean_) * sw)              # PCA in the √w-scaled space
        self.components_ = pca.components_ / sw      # unscale modes back to IV space
        self.k = self.components_.shape[0]
        self.evr_ = pca.explained_variance_ratio_
        self._build_interps()
        return self

    def _build_interps(self) -> None:
        loadings = self.components_.T                # (n_total, k)
        gp = self.grid.grid_pts
        self._comp_interps, self._mean_interps = [], []
        for a in range(self.grid.n_assets):
            s, e = a * self.grid.n_grid, (a + 1) * self.grid.n_grid
            load_a, mean_a = loadings[s:e], self.mean_[s:e]
            self._comp_interps.append((LinearNDInterpolator(gp, load_a),
                                       NearestNDInterpolator(gp, load_a)))
            self._mean_interps.append((LinearNDInterpolator(gp, mean_a),
                                       NearestNDInterpolator(gp, mean_a)))

    # ── transforms ─────────────────────────────────────────────────────────────
    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.where(np.isnan(X), self.mean_, X) - self.mean_
        w = getattr(self, "w_col", None)
        # weighted projection so z is consistent with the weighted-optimal modes (z = (Xc⊙w)Vᵀ);
        # reduces to the ordinary projection when w≡1.
        return (Xc if w is None else Xc * w) @ self.components_.T

    # ── evaluation at arbitrary points ──────────────────────────────────────────
    def observation_matrix(self, feats, aids) -> np.ndarray:
        """H: (N, k) — basis loadings at arbitrary (lm, T) query points."""
        feats = np.asarray(feats, float); aids = np.asarray(aids)
        H = np.zeros((len(feats), self.k), dtype=np.float64)
        for a in range(self.grid.n_assets):
            mask = aids == a
            if not mask.any():
                continue
            pts = feats[mask]
            lin, nn = self._comp_interps[a]
            h = lin(pts)
            bad = np.isnan(h).any(axis=1)
            if bad.any():
                h[bad] = nn(pts[bad])
            H[mask] = h
        return H

    def mean_at(self, feats, aids) -> np.ndarray:
        """Mean surface at arbitrary query points → (N,)."""
        feats = np.asarray(feats, float); aids = np.asarray(aids)
        mu = np.zeros(len(feats), dtype=np.float64)
        for a in range(self.grid.n_assets):
            mask = aids == a
            if not mask.any():
                continue
            pts = feats[mask]
            lin, nn = self._mean_interps[a]
            v = lin(pts)
            bad = np.isnan(v)
            if bad.any():
                v[bad] = nn(pts[bad])
            mu[mask] = v
        return mu


def grid_from_dataset(dataset, n_lm: int = 8, n_T: int | None = None) -> FixedGrid:
    """Build a FixedGrid spanning a dataset's observed (lm, T) range.

    Maturities default to the dataset's unique T values (small discrete grid for the
    Heston DGP); otherwise an evenly spaced set of n_T points.
    """
    qf = dataset.query_feats
    valid = qf[:, :, 1] > 0
    lm = qf[:, :, 0][valid]
    T = qf[:, :, 1][valid]
    lm_lo, lm_hi = np.percentile(lm, [2, 98])
    lm_vals = np.linspace(lm_lo, lm_hi, n_lm)
    uniqT = np.unique(np.round(T, 6))
    if n_T is None and len(uniqT) <= 16:
        T_vals = uniqT
    else:
        T_lo, T_hi = np.percentile(T, [2, 98])
        T_vals = np.linspace(T_lo, T_hi, n_T or 7)
    return FixedGrid(lm_vals, T_vals, dataset.n_assets)
