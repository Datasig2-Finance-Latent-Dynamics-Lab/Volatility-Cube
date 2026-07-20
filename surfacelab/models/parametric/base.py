"""
Per-day parametric surface models (data-only, no training, no temporal prior).

Two flavours, both looping over the assets present on a day:

  * **smile** models (SVI, B-spline) fit one 1-D curve per (asset, maturity) and
    interpolate across maturities in total-variance space for query points whose
    maturity has too few context observations to fit on its own.
  * **surface** models (SSVI) fit one 2-D surface per asset over all (k, T) at once.

The PCA model is separate: it has a real `train()` (it must learn a factor basis).
"""
from __future__ import annotations

import numpy as np

from surfacelab.core.model import SurfaceModel
from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction
from surfacelab.models.parametric import representations as rep


# ── cross-maturity smile evaluation ──────────────────────────────────────────
def _interp_smiles(states_by_T: dict, k_q: np.ndarray, T_q: np.ndarray) -> np.ndarray:
    """Evaluate per-maturity smile states at scattered (k_q, T_q), interpolating w in T."""
    T_sorted = np.sort(np.array(list(states_by_T.keys()), dtype=float))
    out = np.empty(len(k_q), dtype=float)
    for T_i in np.unique(T_q):
        mask = T_q == T_i
        kk = k_q[mask]
        idx = int(np.searchsorted(T_sorted, T_i))
        if idx == 0:
            iv = states_by_T[T_sorted[0]].implied_vol(kk)
        elif idx >= len(T_sorted):
            iv = states_by_T[T_sorted[-1]].implied_vol(kk)
        elif T_sorted[idx] == T_i:
            iv = states_by_T[T_sorted[idx]].implied_vol(kk)
        else:
            T_lo, T_hi = T_sorted[idx - 1], T_sorted[idx]
            wgt = float((T_i - T_lo) / (T_hi - T_lo))
            w_lo = states_by_T[T_lo].total_variance(kk)
            w_hi = states_by_T[T_hi].total_variance(kk)
            w = (1.0 - wgt) * w_lo + wgt * w_hi
            iv = np.sqrt(np.maximum(w, 1e-12) / max(float(T_i), 1e-12))
        out[mask] = iv
    return out


class ParametricSurfaceModel(SurfaceModel):
    """Base for data-only per-day parametric fitters."""

    kind = "smile"          # "smile" or "surface"
    min_pts = 4             # minimum context points to attempt a fit
    t_round = 6             # decimals for grouping maturities

    def _fit_smile(self, k, iv, T):              # → smile state
        raise NotImplementedError

    def _fit_surface(self, k, T, iv):            # → surface state with implied_vol(k,T)
        raise NotImplementedError

    # ── per-asset predictor ───────────────────────────────────────────────────
    def _asset_predictor(self, k, T, iv):
        """Return f(k_q, T_q) -> iv for one asset given its context observations."""
        if k.size < self.min_pts:
            mean_iv = float(np.mean(iv)) if k.size else 0.2
            return lambda kq, Tq: np.full(len(kq), mean_iv)

        if self.kind == "surface":
            state = self._fit_surface(k, T, iv)
            return lambda kq, Tq: np.maximum(state.implied_vol(kq, Tq), 1e-8)

        # smile: fit one curve per maturity with enough points
        Tk = np.round(T, self.t_round)
        states = {}
        for tv in np.unique(Tk):
            m = Tk == tv
            if m.sum() >= self.min_pts:
                try:
                    states[float(tv)] = self._fit_smile(k[m], iv[m], float(tv))
                except Exception:
                    pass
        if not states:
            mean_iv = float(np.mean(iv))
            return lambda kq, Tq: np.full(len(kq), mean_iv)
        return lambda kq, Tq: np.maximum(_interp_smiles(states, kq, Tq), 1e-8)

    # ── contract ───────────────────────────────────────────────────────────────
    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        out = np.empty(query.n, dtype=float)
        for a in np.unique(query.asset_id):
            qm = query.asset_id == a
            cm = context.asset_id == a
            f = self._asset_predictor(context.k[cm], context.T[cm], context.iv[cm])
            out[qm] = f(query.k[qm], query.T[qm])
        return SurfacePrediction(iv=out)


# ── concrete data-only models ────────────────────────────────────────────────
class SVIModel(ParametricSurfaceModel):
    name = "svi"
    kind = "smile"
    min_pts = 5

    def __init__(self, jump_wing: bool = False):
        self.jump_wing = jump_wing

    def _fit_smile(self, k, iv, T):
        return rep.fit_svi_jw(k, iv, T) if self.jump_wing else rep.fit_svi(k, iv, T)


class BSplineModel(ParametricSurfaceModel):
    name = "bspline"
    kind = "smile"
    min_pts = 4

    def _fit_smile(self, k, iv, T):
        return rep.fit_bspline(k, iv, T)


class SSVIModel(ParametricSurfaceModel):
    name = "ssvi"
    kind = "surface"
    min_pts = 6

    def _fit_surface(self, k, T, iv):
        return rep.fit_ssvi(k, T, iv)
