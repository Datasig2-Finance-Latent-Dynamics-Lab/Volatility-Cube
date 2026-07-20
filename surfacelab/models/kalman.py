"""
Kalman factor model: a joint multi-asset PCA basis with AR(1) factor dynamics.

`train` learns the `FactorBasis` from training surfaces, fits per-factor AR(1) dynamics
(A, Q) on the factor time series, and warm-starts the filter state by forward-filtering
the training period.  `seed_prior` projects a full day's surface to factors (the perfect
prior); `predict` runs one Kalman predict + update on today's context and reconstructs
the surface (with factor-uncertainty std); `step` additionally carries the posterior
state forward.

Ported from `KalmanFilter/{kalman,model,fit}.py`; the PCA basis is the shared
`surfacelab.models.factors.FactorBasis`.
"""
from __future__ import annotations

import numpy as np

from surfacelab.core.model import SurfaceModel
from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction
from surfacelab.models.factors import FactorBasis, grid_from_dataset

_IV_FLOOR = 1e-8            # IV positivity floor
_INV_JITTER = 1e-10        # diagonal jitter before inverting a near-singular covariance
_SYM_JITTER = 1e-8         # diagonal jitter when symmetrizing a covariance


def _inv_jit(M, eps=_INV_JITTER):
    """Invert M with a tiny diagonal jitter (A is near a unit root, so M is near-singular)."""
    return np.linalg.inv(M + eps * np.eye(M.shape[0]))


def _symmetrize(M, eps=_SYM_JITTER):
    """Force-symmetric PSD-ish covariance: (M + Mᵀ)/2 plus a diagonal floor."""
    return (M + M.T) / 2 + np.eye(M.shape[0]) * eps


# ── filter core (ported from KalmanFilter/kalman.py) ─────────────────────────
class _KF:
    def __init__(self, A, Q, obs_sigma):
        self.A, self.Q, self.obs_sigma = A, Q, obs_sigma

    def predict(self, z, P):
        return self.A @ z, self.A @ P @ self.A.T + self.Q

    def update(self, z_pred, P_pred, y, H, w=None):
        """Information-form update. `w` is an optional per-observation weight (precision
        multiplier): R⁻¹ = diag(w)/σ². Up-weighting an asset's points lets its few quotes
        carry as much as many peer quotes when estimating the shared factor scores."""
        if len(y) == 0:
            return z_pred, P_pred
        prec = (np.ones(len(y)) if w is None else np.asarray(w, float)) / self.obs_sigma ** 2
        J = _inv_jit(P_pred)                                          # jitter: A is near-unit-root
        HtW = H.T * prec                                              # (k, n_obs)
        J_post = J + HtW @ H
        j_post = J @ z_pred + HtW @ y
        P_post = np.linalg.inv(J_post)
        return P_post @ j_post, P_post


def fit_transition(Z, transition_type="diagonal", ridge_alpha=1.0):
    """Estimate AR(1) (A, Q) from factor time series Z: (n_days, k)."""
    Z_prev, Z_curr = Z[:-1], Z[1:]
    k = Z.shape[1]
    if transition_type == "diagonal":
        denom = (Z_prev ** 2).sum(axis=0) + 1e-12
        a_diag = np.clip((Z_curr * Z_prev).sum(axis=0) / denom, -0.999, 0.999)
        A = np.diag(a_diag)
    else:
        from sklearn.linear_model import Ridge
        # Standardize each state dim to unit scale BEFORE the ridge so a single `alpha`
        # penalises every parameter equally — otherwise small-scale SSVI params (v0~0.04)
        # are crushed toward zero while large ones (kappa~1) are barely touched, leaving an
        # effectively static transition.  Scale only (no centering) keeps z_pred = A z
        # linear with no intercept, matching the Kalman predict step.
        s = Z.std(axis=0)
        s = np.where(s < 1e-8, 1.0, s)
        Zs_prev, Zs_curr = Z_prev / s, Z_curr / s
        reg = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(Zs_prev, Zs_curr)
        A = reg.coef_ * (s[:, None] / s[None, :])      # map standardized A back to raw space
        # Guarantee a stable filter: estimated VARs on near-random-walk vol params sit right
        # at a unit root (spectral radius ≈ 1), which makes the predict-step covariance grow
        # unbounded and the stationary-covariance Lyapunov solve non-PSD.  Scale A down to a
        # spectral radius of 0.99 — keeps the near-persistent dynamics, drops the explosive edge.
        sr = np.max(np.abs(np.linalg.eigvals(A)))
        if sr >= 0.99:
            A = A * (0.99 / sr)
    resid = Z_curr - Z_prev @ A.T
    Q = np.cov(resid.T)
    if Q.ndim == 0:
        Q = np.array([[float(Q)]])
    return A, _symmetrize(Q)


# ── shared transition utilities (used by every Kalman state representation) ───────
def zero_cross_blocks(M: np.ndarray, n_assets: int, block: int) -> np.ndarray:
    """Keep the per-asset `block`×`block` diagonal blocks of M, zero the cross-asset rest.
    The no-cross-asset ablation for any state laid out as per-asset coefficient blocks."""
    out = np.zeros_like(M)
    for a in range(n_assets):
        s = slice(a * block, (a + 1) * block)
        out[s, s] = M[s, s]
    return out


def augment_increments(A_inc: np.ndarray, Q_inc: np.ndarray):
    """Augment an increment AR(1) (Δz_t = A_inc Δz_{t-1}) into a level+increment state
    s = [z; Δz] with z_t = z_{t-1} + A_inc Δz_{t-1}:

        F = [[I, A_inc], [0, A_inc]],   Q_aug = [[Q, Q], [Q, Q]] (shared shock on both blocks).

    Returns (A_aug, Q_aug); the observation sees only the level block z."""
    n = A_inc.shape[0]
    I = np.eye(n)
    A_aug = np.block([[I, A_inc], [np.zeros((n, n)), A_inc]])
    Q_aug = np.block([[Q_inc, Q_inc], [Q_inc, Q_inc]]) + 1e-8 * np.eye(2 * n)
    return A_aug, Q_aug


# ── the model ─────────────────────────────────────────────────────────────────
class KalmanFactorModel(SurfaceModel):
    """Kalman filter on the coefficients of a LINEAR surface basis.

    The state is a coefficient vector mapped to IV by a fixed linear basis (iv ≈ mean + H z),
    so the whole filter is closed-form linear.  Two axes generalise it beyond the original
    PCA-levels model:

      * ``basis_kind`` — ``"pca"`` (joint multi-asset modes, FactorBasis) or ``"bspline"``
        (fixed tensor B-spline coefficients, per-asset blocks, BSplineBasis).
      * ``transition_mode`` — ``"levels"`` (AR(1) on the coefficients, z_t = A z_{t-1}) or
        ``"increments"`` (AR(1) on Δz, augmented state [z; Δz]) — the SSVI-Kalman trick
        applied to a richer linear representation.

    ``cross_asset=False`` zeroes the cross-asset blocks of the learned transition (and its
    process noise), the no-cross-asset ablation.  It needs a per-asset block layout, so it is
    only meaningful for the B-spline basis; PCA modes mix assets and stay cross-asset.
    """

    def __init__(self, n_components: int = 20, obs_sigma: float = 0.01,
                 p0_scale: float = 5.0, transition: str = "diagonal",
                 weight_asset: int | None = None, asset_weight: float = 1.0,
                 basis_kind: str = "pca", transition_mode: str = "levels",
                 cross_asset: bool = True, ridge_alpha: float = 1.0,
                 n_history: int | None = None, n_lm: int = 6, n_T: int = 4):
        if basis_kind not in ("pca", "bspline"):
            raise ValueError(f"basis_kind must be 'pca'|'bspline', got {basis_kind!r}")
        if transition_mode not in ("levels", "increments"):
            raise ValueError(f"transition_mode must be 'levels'|'increments', got {transition_mode!r}")
        if basis_kind == "pca" and not cross_asset:
            raise ValueError("cross_asset=False needs per-asset coefficient blocks; use basis_kind='bspline'")
        self.n_components = n_components
        self.obs_sigma = obs_sigma
        self.p0_scale = p0_scale
        self.transition = transition
        self.basis_kind = basis_kind
        self.transition_mode = transition_mode
        self.cross_asset = cross_asset
        self.ridge_alpha = ridge_alpha
        self.n_history = n_history
        self.n_lm, self.n_T = n_lm, n_T
        # focus the PCA basis on one asset: that asset's grid columns get `asset_weight`
        # (others 1) in a weighted PCA, so the modes reconstruct it preferentially.
        self.weight_asset = weight_asset
        self.asset_weight = asset_weight
        # focus the per-day FILTER UPDATE on one asset: its observations get `obs_weight`x
        # precision (others 1). Set after train(); does NOT affect the basis.
        self.obs_weight_asset = None
        self.obs_weight = 1.0
        self.basis = None
        self.kf: _KF | None = None
        self.z_warm = self.P_warm = None
        self._z = self._P = None
        self._cpa: int | None = None          # coeffs per asset (B-spline; None for PCA)
        self._aug = (transition_mode == "increments")
        self._Hgrid = self._mean_grid = None  # cached grid observation matrix / mean
        self.name = self._make_name()

    def _make_name(self) -> str:
        base = "kalman_pca" if self.basis_kind == "pca" else "kalman_bspline"
        if self.transition_mode == "increments":
            base += "_inc"
        return base if self.cross_asset else base + "_nox"

    # ── training ────────────────────────────────────────────────────────────────
    def train(self, data, *, saved: bool = False, force: bool = False) -> None:
        self.data_tag = data.meta.get("dgp", self.data_tag)
        grid = grid_from_dataset(data)
        train_idx = data.train_idx()
        if self.n_history and len(train_idx) > self.n_history:
            train_idx = train_idx[-self.n_history:]
        X = grid.stack_days(data, train_idx)

        if self.basis_kind == "bspline":
            from surfacelab.models.bspline_basis import BSplineBasis
            self.basis = BSplineBasis(grid, n_lm=self.n_lm, n_T=self.n_T).fit(X)
            self._cpa = self.basis.coeffs_per_asset
        else:
            w = None
            if self.weight_asset is not None:
                w = np.ones(grid.n_total)
                s = self.weight_asset * grid.n_grid
                w[s:s + grid.n_grid] = self.asset_weight
            self.basis = FactorBasis(self.n_components, grid).fit(X, weights=w)
        Z = self.basis.transform(X)
        k = self.basis.k

        # learn the transition on levels or on increments, with optional cross-asset zeroing
        src = Z if not self._aug else np.diff(Z, axis=0)
        A_core, Q_core = fit_transition(src, transition_type=self.transition,
                                        ridge_alpha=self.ridge_alpha)
        if not self.cross_asset:
            A_core = zero_cross_blocks(A_core, grid.n_assets, self._cpa)
            Q_core = zero_cross_blocks(Q_core, grid.n_assets, self._cpa)
        if self._aug:
            A, Q = augment_increments(A_core, Q_core)
        else:
            A, Q = A_core, Q_core
        self.kf = _KF(A, Q, self.obs_sigma)

        # cache the grid observation matrix once (basis-agnostic; PCA returns its modes,
        # B-spline its tensor design) so the warm-up filter is representation-independent.
        n_assets, ng = grid.n_assets, grid.n_grid
        gfeats = np.tile(grid.grid_pts, (n_assets, 1))
        gaids = np.repeat(np.arange(n_assets), ng)
        self._Hgrid = self.basis.observation_matrix(gfeats, gaids)
        self._mean_grid = self.basis.mean_at(gfeats, gaids)

        # warm-up forward filter over training (state dim doubles in increments mode)
        d = 2 * k if self._aug else k
        z = np.zeros(d)
        P = np.eye(d) * self.obs_sigma ** 2 * self.p0_scale
        for i in range(len(train_idx)):
            z, P = self._step_filter(z, P, X[i])
        self.z_warm, self.P_warm = z.copy(), P.copy()
        self._z, self._P = z.copy(), P.copy()

    def _obs(self, H: np.ndarray) -> np.ndarray:
        """Observation matrix into the (possibly augmented) state: levels see z directly;
        increments observe only the level block, so the Δz columns are zero."""
        return H if not self._aug else np.hstack([H, np.zeros_like(H)])

    def _step_filter(self, z, P, grid_surface):
        """Predict+update from a full grid surface (used during warm-up)."""
        z_pred, P_pred = self.kf.predict(z, P)
        valid = ~np.isnan(grid_surface)
        if not valid.any():
            return z_pred, P_pred
        H = self._obs(self._Hgrid[valid])
        y = grid_surface[valid] - self._mean_grid[valid]
        return self.kf.update(z_pred, P_pred, y, H)

    # ── prior / sequence ──────────────────────────────────────────────────────
    def reset_sequence(self) -> None:
        if self.z_warm is not None:
            self._z, self._P = self.z_warm.copy(), self.P_warm.copy()

    def seed_prior(self, quotes: Quotes) -> None:
        """Project a full day's surface to coefficients → the seed state.  In increments mode
        the increment block is seeded at 0 (an unknowable trend from one day would leak)."""
        grid = self.basis.grid
        surf = grid.interp_day(quotes.feats, quotes.asset_id, quotes.iv).ravel()
        z_lvl = self.basis.transform(surf[None, :])[0]
        self._z = np.concatenate([z_lvl, np.zeros(self.basis.k)]) if self._aug else z_lvl
        self._P = self.P_warm.copy() if self.P_warm is not None \
            else np.eye(self._z.size) * self.obs_sigma ** 2 * self.p0_scale

    # ── prediction ───────────────────────────────────────────────────────────────
    def _filter_today(self, context: Quotes):
        z_pred, P_pred = self.kf.predict(self._z, self._P)
        if context.n:
            H = self._obs(self.basis.observation_matrix(context.feats, context.asset_id))
            y = context.iv - self.basis.mean_at(context.feats, context.asset_id)
            w = None
            if self.obs_weight_asset is not None and self.obs_weight != 1.0:
                w = np.where(context.asset_id == self.obs_weight_asset, self.obs_weight, 1.0)
            return self.kf.update(z_pred, P_pred, y, H, w=w)
        return z_pred, P_pred

    def _reconstruct(self, z, P, query: QueryPoints) -> SurfacePrediction:
        H = self._obs(self.basis.observation_matrix(query.feats, query.asset_id))
        mu = self.basis.mean_at(query.feats, query.asset_id)
        iv = H @ z + mu
        var = np.einsum("ij,jk,ik->i", H, P, H)
        return SurfacePrediction(iv=np.maximum(iv, _IV_FLOOR),
                                 iv_std=np.sqrt(np.maximum(var, 0.0)))

    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self.basis is None:
            raise RuntimeError("KalmanFactorModel.predict before train()")
        z, P = self._filter_today(context)
        return self._reconstruct(z, P, query)

    def step(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        z, P = self._filter_today(context)
        self._z, self._P = z, P              # carry posterior forward
        return self._reconstruct(z, P, query)
