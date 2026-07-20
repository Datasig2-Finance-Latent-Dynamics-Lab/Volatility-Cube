"""
Kalman filter on the *stacked* SSVI parameter vector across all assets.

Where `kalman_pca` (KalmanFactorModel) runs the filter on a PCA grid, this runs it on
the SSVI parameters themselves.  The state is the concatenation of every asset's six
SSVI parameters,

    z = [ (v0,v∞,κ,ρ,η,γ)_asset0 , (…)_asset1 , … ]   ∈ R^{6N},

so the learned transition matrix `A` is a *single, general* 6N×6N map — NOT a per-asset
block — and its off-diagonal blocks are exactly the cross-asset dependencies (yesterday's
SPY parameters informing today's AAPL, etc.).

It behaves like the temporal/graph SSVI models (`ssvi_temporal[_graph]`) but pulls today's
fit toward the *transformed* prior `A·z_prev` instead of the raw prior `z_prev`:

    z_pred = A z_prev,                P_pred = A P Aᵀ + Q          (Kalman predict)
    θ* = argmin_θ  ‖w_fit(θ) − w_obs‖²/σ²  +  (θ − z_pred)ᵀ P_pred⁻¹ (θ − z_pred)
    P_post = ( P_pred⁻¹ + HᵀH/σ² )⁻¹,   H = ∂w_ctx/∂θ |_{θ*}        (EKF update)

The mean update is the MAP estimate (SSVI is non-linear in its parameters, so the update
can't be the linear KF formula); the covariance update is the standard EKF/Gauss-Newton
information form.  `A`, `Q` are learned from the SSVI-parameter time series produced by
fitting SSVI to every (asset, day) — which is why a fast warm-started fitter is needed.

`A` carries all cross-asset structure, so this model uses NO graph-edge machinery.
"""
from __future__ import annotations

import numpy as np

from surfacelab.core.model import SurfaceModel
from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction, query_as_quotes
from surfacelab.models.kalman import (fit_transition, _inv_jit, _symmetrize,
                                       zero_cross_blocks, augment_increments)
from surfacelab.models.parametric.representations import (
    fit_ssvi_fast, ssvi_tv_grad, _SSVI_BOUNDS)

_D = 6  # SSVI params per asset
_IV_FLOOR = 1e-8           # IV positivity floor
_DEFAULT_IV = 0.2          # flat-vol fallback for an asset with no params
_COV_FLOOR = 1e-6          # diagonal floor added to a warm/empirical covariance


def _ffill_bfill(Z: np.ndarray) -> np.ndarray:
    """Fill NaN rows-of-columns forward then backward (days an asset had too few quotes)."""
    Z = Z.copy()
    for j in range(Z.shape[1]):
        col = Z[:, j]
        idx = np.where(~np.isnan(col))[0]
        if idx.size == 0:
            col[:] = 0.0
            continue
        # forward fill
        last = col[idx[0]]
        for i in range(Z.shape[0]):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]
        # back fill the leading gap
        col[:idx[0]] = col[idx[0]]
    return Z


class KalmanSSVIModel(SurfaceModel):
    name = "kalman_ssvi"

    def __init__(self, obs_sigma: float = 0.01, ridge_alpha: float = 1.0,
                 n_history: int = 250, maxiter: int = 150,
                 transition_mode: str = "levels", cross_asset: bool = True):
        if transition_mode not in ("levels", "increments"):
            raise ValueError(f"transition_mode must be 'levels'|'increments', got {transition_mode}")
        self.obs_sigma = obs_sigma
        self.ridge_alpha = ridge_alpha
        self.n_history = n_history
        self.maxiter = maxiter
        self.transition_mode = transition_mode
        # cross_asset=False → block-diagonal A (own-asset dynamics only): the ablation that
        # isolates whether the learned cross-asset coupling actually improves prediction.
        self.cross_asset = cross_asset
        base = "kalman_ssvi" if transition_mode == "levels" else "kalman_ssvi_inc"
        self.name = base if cross_asset else base + "_nox"
        self.n_assets: int | None = None
        self.A = self.Q = self.Q_aug = None   # transition + process-noise (augmented in increments)
        self.A_core = None              # the learned per-step coupling (for inspection/heatmap)
        self.z_warm = self.P_warm = None
        self.z_mean = None              # neutral training-mean level (non-leaking seed fallback)
        self._z = self._P = None
        self._n = None                  # core state dim (6N); augmented dim is 2*_n in increments

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, data, *, saved: bool = False, force: bool = False) -> None:
        self.data_tag = data.meta.get("dgp", self.data_tag)
        N = self.n_assets = int(data.n_assets)
        train_idx = data.train_idx()
        if self.n_history and len(train_idx) > self.n_history:
            train_idx = train_idx[-self.n_history:]

        # bulk SSVI fits over every (asset, day), warm-started day-over-day per asset
        Z = np.full((len(train_idx), N * _D), np.nan)
        last: dict[int, np.ndarray | None] = {a: None for a in range(N)}
        for i, t in enumerate(train_idx):
            q = data.quotes_at(t)
            for a in range(N):
                sel = q.asset_id == a
                if sel.sum() >= _D:
                    st = fit_ssvi_fast(q.k[sel], q.T[sel], q.iv[sel],
                                       x0=last[a], maxiter=self.maxiter)
                    p = np.array([st.v_0, st.v_inf, st.kappa, st.rho, st.eta, st.gamma])
                    last[a] = p
                    Z[i, a * _D:(a + 1) * _D] = p
        Z = _ffill_bfill(Z)
        self._n = N * _D
        # Generic "typical surface" prior: the mean SSVI level over training days.  Used as the
        # seed fallback for an asset with too few context quotes to fit — a fair learned prior,
        # NOT the specific last-training-day surface (self.z_warm), which would leak a near-
        # perfect prior whenever the context budget is small.
        self.z_mean = Z.mean(axis=0)

        if self.transition_mode == "levels":
            # full (cross-asset) AR(1) on the parameter LEVELS: z_t = A z_{t-1}
            self.A, self.Q = fit_transition(Z, transition_type="full",
                                             ridge_alpha=self.ridge_alpha)
            if not self.cross_asset:
                # truly isolate own-asset dynamics: zero BOTH the transition coupling and
                # the correlated process noise (else peers' shocks still leak via Q in P_pred).
                self.A = zero_cross_blocks(self.A, N, _D)
                self.Q = zero_cross_blocks(self.Q, N, _D)
            self.A_core = self.A
            self.z_warm = Z[-1].copy()
            self.P_warm = self._stationary_cov(Z)
        else:
            # full (cross-asset) AR(1) on the INCREMENTS: Δz_t = A Δz_{t-1}.
            # Augmented state s = [z; Δz]; the cross-asset coupling A is preserved in full.
            #   z_t  = z_{t-1} + A Δz_{t-1}      Δz_t = A Δz_{t-1}
            #   F = [[I, A], [0, A]]             noise ε hits both blocks → Q_aug=[[Q,Q],[Q,Q]]
            DZ = np.diff(Z, axis=0)
            A_inc, Q_inc = fit_transition(DZ, transition_type="full",
                                          ridge_alpha=self.ridge_alpha)
            if not self.cross_asset:
                A_inc = zero_cross_blocks(A_inc, N, _D)
                Q_inc = zero_cross_blocks(Q_inc, N, _D)
            self.A_core, self.Q = A_inc, Q_inc
            n = self._n
            self.A, self.Q_aug = augment_increments(A_inc, Q_inc)
            # warm state: last level + last increment; warm cov block-diag (levels var, incr cov)
            last_inc = Z[-1] - Z[-2] if len(Z) >= 2 else np.zeros_like(Z[-1])
            self.z_warm = np.concatenate([Z[-1], last_inc])
            P_lvl = np.diag(np.var(Z, axis=0) + _COV_FLOOR)
            P_inc = _symmetrize(Q_inc, eps=_COV_FLOOR)
            self.P_warm = np.block([[P_lvl, np.zeros((n, n))],
                                    [np.zeros((n, n)), P_inc]])
        self._z, self._P = self.z_warm.copy(), self.P_warm.copy()

    def _stationary_cov(self, Z: np.ndarray) -> np.ndarray:
        """Stationary state covariance P = A P Aᵀ + Q (the prior uncertainty at seed time).

        Falls back to the empirical parameter covariance if the learned A is not stable.
        """
        try:
            from scipy.linalg import solve_discrete_lyapunov
            P = solve_discrete_lyapunov(self.A, self.Q)
            if np.all(np.isfinite(P)) and np.all(np.linalg.eigvalsh((P + P.T) / 2) > -1e-8):
                return _symmetrize(P)
        except Exception:
            pass
        return np.diag(np.var(Z, axis=0) + _COV_FLOOR)

    # ── prior / sequence ────────────────────────────────────────────────────────
    def reset_sequence(self) -> None:
        if self.z_warm is not None:
            self._z, self._P = self.z_warm.copy(), self.P_warm.copy()

    def _fit_day(self, quotes: Quotes) -> np.ndarray:
        """Per-asset SSVI fit of the SEEDED quotes → the stacked LEVEL vector.

        Only assets with ≥ _D quotes in `quotes` are fit; the rest fall back to the neutral
        training-mean level (`z_mean`).  Crucially the fallback is NOT the last-training-day
        fit, so a sparse seed cannot smuggle in a full-surface prior."""
        n = self._n
        base = self.z_mean if self.z_mean is not None else np.zeros(self.n_assets * _D)
        z = base.copy()
        for a in range(self.n_assets):
            sel = quotes.asset_id == a
            if sel.sum() >= _D:
                st = fit_ssvi_fast(quotes.k[sel], quotes.T[sel], quotes.iv[sel],
                                   x0=z[a * _D:(a + 1) * _D], maxiter=self.maxiter)
                z[a * _D:(a + 1) * _D] = [st.v_0, st.v_inf, st.kappa, st.rho, st.eta, st.gamma]
        return z

    def seed_prior(self, quotes: Quotes) -> None:
        if self.A is None:
            raise RuntimeError("KalmanSSVIModel.seed_prior before train()")
        z_lvl = self._fit_day(quotes)
        if self.transition_mode == "increments":
            # The increment is unknowable from a single (sparse) seeded day, so seed it at 0
            # rather than the trained end-of-training increment (which would leak a trend).
            self._z = np.concatenate([z_lvl, np.zeros(self._n)])
        else:
            self._z = z_lvl
        self._P = self.P_warm.copy()

    # ── Kalman predict + MAP update ──────────────────────────────────────────────
    def _context_blocks(self, context: Quotes) -> dict:
        """Per-asset context observations (k, T, w_obs) for assets with any quotes today."""
        out = {}
        if context.n:
            for a in range(self.n_assets):
                sel = context.asset_id == a
                if sel.any():
                    out[a] = (context.k[sel], context.T[sel],
                              context.iv[sel] ** 2 * context.T[sel])
        return out

    def _update(self, context: Quotes):
        """One Kalman predict + (non-linear MAP) update; returns (θ*, P_post)."""
        N, D = self.n_assets, _D
        Qmat = self.Q_aug if self.transition_mode == "increments" else self.Q
        z_pred = self.A @ self._z
        P_pred = self.A @ self._P @ self.A.T + Qmat
        d = z_pred.size
        Pinv = _inv_jit(P_pred)
        s2 = self.obs_sigma ** 2
        blocks = self._context_blocks(context)

        def obj_grad(theta):
            r = theta - z_pred
            obj = float(r @ Pinv @ r)
            grad = 2.0 * (Pinv @ r)
            for a, (k, T, w_obs) in blocks.items():
                sl = slice(a * D, (a + 1) * D)
                w, dw = ssvi_tv_grad(theta[sl], k, T)
                resid = w - w_obs
                obj += float(resid @ resid) / s2
                grad[sl] += (2.0 / s2) * (dw * resid).sum(axis=1)
            return obj, grad

        bounds = list(_SSVI_BOUNDS) * N                       # level block: SSVI bounds
        if d > N * D:                                         # increment block: unbounded
            bounds = bounds + [(None, None)] * (d - N * D)
        from scipy.optimize import minimize
        res = minimize(obj_grad, z_pred.copy(), method="L-BFGS-B", jac=True,
                       bounds=bounds, options={"maxiter": self.maxiter})
        theta = res.x

        # EKF / Gauss-Newton covariance update: information form, block-diagonal HᵀH
        info = Pinv.copy()
        for a, (k, T, _) in blocks.items():
            sl = slice(a * D, (a + 1) * D)
            _, dw = ssvi_tv_grad(theta[sl], k, T)        # (D, n_pts)
            info[sl, sl] += (dw @ dw.T) / s2
        P_post = _inv_jit(info)
        return theta, P_post

    def _reconstruct(self, theta, P, query: QueryPoints) -> SurfacePrediction:
        N, D = self.n_assets, _D
        iv = np.full(query.n, _DEFAULT_IV)
        var = np.zeros(query.n)
        for a in np.unique(query.asset_id):
            qm = query.asset_id == a
            sl = slice(int(a) * D, (int(a) + 1) * D)
            p = theta[sl]
            k, T = query.k[qm], query.T[qm]
            w, dw = ssvi_tv_grad(p, k, T)
            ivq = np.sqrt(np.maximum(w / np.maximum(T, 1e-12), 1e-12))
            iv[qm] = np.maximum(ivq, _IV_FLOOR)
            # iv = sqrt(w/T)  ⇒  ∂iv/∂θ = ∂w/∂θ / (2 T iv);  var = gᵀ P_block g
            G = dw / (2.0 * T * np.maximum(ivq, _IV_FLOOR))    # (D, n_pts)
            var[qm] = np.einsum("jn,jk,kn->n", G, P[sl, sl], G)
        return SurfacePrediction(iv=iv, iv_std=np.sqrt(np.maximum(var, 0.0)))

    # ── contract ──────────────────────────────────────────────────────────────
    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self.A is None:
            raise RuntimeError("KalmanSSVIModel.predict before train()")
        if self._z is None:
            self.seed_prior(context if context.n else query_as_quotes(query))
        theta, P_post = self._update(context)
        return self._reconstruct(theta, P_post, query)

    def step(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self._z is None:
            self.seed_prior(context)
        theta, P_post = self._update(context)
        self._z, self._P = theta, P_post              # carry posterior forward
        return self._reconstruct(theta, P_post, query)
