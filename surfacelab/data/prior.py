from __future__ import annotations
import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline
from surfacelab.core.types import Quotes
from surfacelab.data.dataset import Dataset as SurfaceDataset
from surfacelab.models.factors import interp_linear_nearest

_FALLBACK_IV = 0.2          # flat-vol fallback when an asset has no usable quotes
_PRIOR_IV_CAP = 10.0        # IV above 1000% is garbage → drop to NaN
_IV_FLOOR = 1e-8            # IV positivity floor
_MAX_INTERIOR_KNOTS = 5     # cap on per-smile interior B-spline knots


def _clip_prior(prior: np.ndarray) -> np.ndarray:
    """NaN out absurd (>_PRIOR_IV_CAP) prior IVs in place; return the array."""
    finite = np.isfinite(prior)
    prior[finite & (prior > _PRIOR_IV_CAP)] = np.nan
    return prior


def compute_linear_prior(dataset: SurfaceDataset) -> np.ndarray:
    """Linear-interpolation analogue of `compute_bspline_prior`: for each day t >= 1, linearly
    interpolate day t-1's full surface (per asset, in (k, T)) and evaluate at day t's query
    points (nearest-neighbour fallback outside the convex hull).  The market grid is dense, so
    plain linear interpolation is an accurate, cheap prior.  Returns (N_days, N_points), NaN
    for day 0; chronological order assumed (do NOT subsample days)."""
    qf, aid, tgt = dataset.query_feats, dataset.asset_ids, dataset.targets
    N_days, N_points = tgt.shape
    prior = np.full((N_days, N_points), np.nan, dtype=np.float32)
    for t in range(1, N_days):
        vp = qf[t - 1, :, 1] > 0
        vt = qf[t, :, 1] > 0
        for a in range(dataset.n_assets):
            mp = vp & (aid[t - 1] == a)
            mt = vt & (aid[t] == a)
            if mt.sum() == 0:
                continue
            pts = qf[t - 1, mp, :2]
            vals = tgt[t - 1, mp]
            q = qf[t, mt, :2]
            if mp.sum() < 3:
                prior[t, mt] = float(np.mean(vals)) if mp.sum() else _FALLBACK_IV
                continue
            prior[t, mt] = interp_linear_nearest(pts, vals, q).astype(np.float32)
    return _clip_prior(prior)


def compute_bspline_prior(dataset: SurfaceDataset, degree: int = 3) -> np.ndarray:
    """For each day t >= 1, fit per-asset per-maturity B-splines to day t-1's full
    observations and evaluate them at day t's query points.

    Returns prior_targets of shape (N_days, N_points) float32, NaN for day 0.  Assumes
    days are in chronological order.  Do NOT subsample days when using this — the prior
    for day t must come from the actual preceding trading day in the array.
    """
    N_days, N_points = dataset.targets.shape
    prior = np.full((N_days, N_points), np.nan, dtype=np.float32)

    qf  = dataset.query_feats  # (N_days, N_points, 2) = (lm, T)
    aid = dataset.asset_ids
    tgt = dataset.targets

    for t in range(1, N_days):
        if t % 100 == 0:
            print(f"  Prior: day {t}/{N_days - 1}", end="\r", flush=True)

        valid_prev = qf[t - 1, :, 1] > 0
        lm_prev  = qf[t - 1, :, 0][valid_prev]
        T_prev   = qf[t - 1, :, 1][valid_prev]
        aid_prev = aid[t - 1][valid_prev]
        iv_prev  = tgt[t - 1][valid_prev]

        valid_t = qf[t, :, 1] > 0
        lm_t  = qf[t, :, 0]
        T_t   = qf[t, :, 1]
        aid_t = aid[t]

        for a in range(dataset.n_assets):
            mask_prev = aid_prev == a
            if mask_prev.sum() < degree + 1:
                continue

            lm_a = lm_prev[mask_prev]
            T_a  = T_prev[mask_prev]
            iv_a = iv_prev[mask_prev]

            unique_T = np.unique(T_a)
            smile_by_T: dict = {}
            for T_j in unique_T:
                mask_T = T_a == T_j
                smile_by_T[T_j] = _fit_smile(lm_a[mask_T], iv_a[mask_T], degree)

            mask_t_a = valid_t & (aid_t == a)
            if not mask_t_a.any():
                continue

            prior[t, mask_t_a] = _eval_surface(
                smile_by_T, unique_T, lm_t[mask_t_a], T_t[mask_t_a])

    if N_days > 1:
        print(f"  Prior: done ({N_days - 1} days)           ")

    return _clip_prior(prior)


def _fit_smile(lm: np.ndarray, iv: np.ndarray, degree: int = 3):
    """Fit a 1-D B-spline to (lm, iv); return a vectorised callable."""
    order = np.argsort(lm)
    lm = lm[order].astype(float)
    iv = iv[order].astype(float)

    if len(lm) < degree + 1:
        return _flat(float(np.mean(iv)))

    k_lo, k_hi = float(lm[0]), float(lm[-1])
    if k_hi <= k_lo + 1e-8:
        return _flat(float(np.mean(iv)))

    n_interior = min(_MAX_INTERIOR_KNOTS, len(lm) - degree - 1)
    if n_interior < 0:
        return _flat(float(np.mean(iv)))

    interior = np.linspace(k_lo, k_hi, n_interior + 2)[1:-1]
    knots = np.concatenate([
        np.repeat(k_lo, degree + 1),
        interior,
        np.repeat(k_hi, degree + 1),
    ])

    try:
        spl = make_lsq_spline(lm, iv, knots, k=degree)
        iv_max = float(iv.max())
        if spl.c.max() > max(iv_max * 3.0, 5.0) or spl.c.min() < -0.1:
            return _flat(float(np.mean(iv)))

        coeffs  = np.maximum(spl.c, _IV_FLOOR)
        spl_obj = BSpline(spl.t, coeffs, spl.k, extrapolate=False)
        fallback = float(np.mean(iv))

        def evaluator(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            x_clip = np.clip(x, k_lo, k_hi)
            out = spl_obj(x_clip)
            bad = ~np.isfinite(out)
            if bad.any():
                out[bad] = fallback
            return np.clip(out, 1e-4, iv_max * 2.0).astype(np.float32)

        return evaluator
    except Exception:
        return _flat(float(np.mean(iv)))


def _flat(val: float):
    val = max(val, _IV_FLOOR)

    def evaluator(x: np.ndarray) -> np.ndarray:
        return np.full(len(np.atleast_1d(np.asarray(x))), val, dtype=np.float32)

    return evaluator


def _eval_surface(smile_by_T: dict, unique_T: np.ndarray,
                  lm_q: np.ndarray, T_q: np.ndarray) -> np.ndarray:
    """Evaluate the per-maturity smiles at scattered (lm_q, T_q) points."""
    T_sorted = np.sort(unique_T)
    result   = np.empty(len(lm_q), dtype=np.float32)

    for T_i in np.unique(T_q):
        mask = T_q == T_i
        lm_slice = lm_q[mask]
        idx = int(np.searchsorted(T_sorted, T_i))

        if idx == 0:
            result[mask] = smile_by_T[T_sorted[0]](lm_slice)
        elif idx >= len(T_sorted):
            result[mask] = smile_by_T[T_sorted[-1]](lm_slice)
        elif T_sorted[idx] == T_i:
            result[mask] = smile_by_T[T_sorted[idx]](lm_slice)
        else:
            T_lo = T_sorted[idx - 1]
            T_hi = T_sorted[idx]
            w    = float((T_i - T_lo) / (T_hi - T_lo))
            iv_lo = smile_by_T[T_lo](lm_slice)
            iv_hi = smile_by_T[T_hi](lm_slice)
            result[mask] = ((1.0 - w) * iv_lo + w * iv_hi).astype(np.float32)

    return result


# ════════════════════════════════════════════════════════════════════════════
# One-day carried prior surfaces (used by the delta-CNP AND the persistence baseline)
# ════════════════════════════════════════════════════════════════════════════
class _PerAssetPrior:
    """Shared scaffold for the one-day carried prior surfaces.

    Holds the per-asset flat-mean `fallback` (used where an asset lacks a usable fit) and an
    `eval` template that dispatches each asset to its fitted `by_asset` entry via the subclass
    hook `_eval_asset`.  Subclasses populate `self.by_asset` in their __init__."""

    def __init__(self, quotes: Quotes):
        self.by_asset: dict[int, object] = {}
        self.fallback: dict[int, float] = {}
        for a in np.unique(quotes.asset_id):
            m = quotes.asset_id == a
            self.fallback[int(a)] = float(np.mean(quotes.iv[m])) if m.any() else _FALLBACK_IV

    def _eval_asset(self, entry, k, T) -> np.ndarray:
        raise NotImplementedError

    def eval(self, k, T, asset_id) -> np.ndarray:
        out = np.empty(len(k), dtype=float)
        for a in np.unique(asset_id):
            qm = asset_id == a
            entry = self.by_asset.get(int(a))
            if entry is None:
                out[qm] = self.fallback.get(int(a), _FALLBACK_IV)
            else:
                out[qm] = self._eval_asset(entry, k[qm], T[qm])
        return np.maximum(out, _IV_FLOOR)


class _BSplinePrior(_PerAssetPrior):
    """One day's B-spline prior surface.

    Built with the *exact same* per-(asset, maturity) machinery as
    `compute_bspline_prior` (adaptive per-smile knots, ill-conditioning rejection,
    clamp-to-data-range, output clipping), so the prior the delta-CNP subtracts at
    inference and the prior the persistence baseline carries are *identical* — and
    robust on the steep short-dated wings where a fixed-knot fit blows up.
    """

    def __init__(self, quotes: Quotes, degree: int = 3):
        super().__init__(quotes)
        self.degree = degree
        # asset -> (smile_by_T, unique_T)
        for a in np.unique(quotes.asset_id):
            m = quotes.asset_id == a
            if m.sum() < degree + 1:
                continue
            T_a = quotes.T[m]
            uniq_T = np.unique(T_a)
            smile_by_T = {Tj: _fit_smile(quotes.k[m][T_a == Tj], quotes.iv[m][T_a == Tj],
                                         degree) for Tj in uniq_T}
            self.by_asset[int(a)] = (smile_by_T, uniq_T)

    def _eval_asset(self, entry, k, T) -> np.ndarray:
        smile_by_T, uniq_T = entry
        return _eval_surface(smile_by_T, uniq_T, k, T)


class _LinearPrior(_PerAssetPrior):
    """One day's linear-interpolation prior surface (per asset, in (k, T); nearest-neighbour
    fallback outside the hull).  The dense market grid makes plain linear interpolation an
    accurate, cheap prior — matches `compute_linear_prior`."""

    def __init__(self, quotes: Quotes):
        super().__init__(quotes)
        # asset -> (pts (N,2), vals (N,))
        for a in np.unique(quotes.asset_id):
            m = quotes.asset_id == a
            if m.sum() >= 3:
                self.by_asset[int(a)] = (np.stack([quotes.k[m], quotes.T[m]], axis=1),
                                         quotes.iv[m])

    def _eval_asset(self, entry, k, T) -> np.ndarray:
        pts, vals = entry
        return interp_linear_nearest(pts, vals, np.stack([k, T], axis=1))
