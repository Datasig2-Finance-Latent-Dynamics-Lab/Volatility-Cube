"""
No-arbitrage diagnostics on a predicted surface.

Two static no-arbitrage conditions, evaluated directly on arrays of predicted IV so
they work for *any* model (not just the parametric ones):

  * **Butterfly** (Durrleman): on each (asset, maturity) smile, the density is
    non-negative iff g(k) >= 0, where
        g = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2,   w = iv^2 * T.
  * **Calendar**: total variance must be non-decreasing in maturity at fixed strike.

Each returns the *fraction of evaluated points that violate* (a percentage), matching
the `butterfly_pct` / `calendar_pct` columns of the eval summary.

Ported from the per-node penalties in
`dgraph/examples/vol_smiles/losses/node.py` (BSplineNALoss, CalendarSpreadPenalty),
but rewritten to act on raw (k, T, iv) arrays.
"""
from __future__ import annotations

import numpy as np

_T_DECIMALS = 6      # round maturities to this many dp when bucketing points into smiles


def total_variance(iv: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.asarray(iv) ** 2 * np.asarray(T)


def durrleman_g(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Durrleman function g(k) on a smile (k must be sorted ascending).

    Returns an array the same length as k; g < 0 flags a butterfly violation.
    Uses np.gradient finite differences (needs >= 3 points).
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if k.size < 3:
        return np.full(k.shape, np.nan)
    wp = np.gradient(w, k)
    wpp = np.gradient(wp, k)
    w_safe = np.maximum(w, 1e-8)
    return (1.0 - k * wp / (2.0 * w_safe)) ** 2 \
        - wp ** 2 / 4.0 * (1.0 / w_safe + 0.25) \
        + wpp / 2.0


def _group_smiles(k, T, iv, asset_id, t_round: int = _T_DECIMALS):
    """Yield (asset, T, k_sorted, iv_sorted) per (asset, maturity) smile."""
    k = np.asarray(k, float)
    T = np.asarray(T, float)
    iv = np.asarray(iv, float)
    asset_id = np.asarray(asset_id)
    Tkey = np.round(T, t_round)
    for a in np.unique(asset_id):
        for tv in np.unique(Tkey[asset_id == a]):
            m = (asset_id == a) & (Tkey == tv)
            if m.sum() < 3:
                continue
            kk = k[m]
            order = np.argsort(kk)
            yield int(a), float(tv), kk[order], iv[m][order]


def butterfly_pct(k, T, iv, asset_id) -> float:
    """Percentage of smile points where the Durrleman density is negative."""
    total = viol = 0
    for _a, tv, kk, ivv in _group_smiles(k, T, iv, asset_id):
        g = durrleman_g(kk, total_variance(ivv, tv))
        g = g[np.isfinite(g)]
        total += g.size
        viol += int(np.sum(g < 0.0))
    return 100.0 * viol / total if total else 0.0


def calendar_pct(k, T, iv, asset_id, n_grid: int = 40) -> float:
    """Percentage of (strike, maturity-pair) checks where total variance decreases.

    For each asset, consecutive maturities are compared on the overlapping
    log-moneyness range (linearly interpolated total variance).
    """
    k = np.asarray(k, float)
    T = np.asarray(T, float)
    iv = np.asarray(iv, float)
    asset_id = np.asarray(asset_id)
    Tkey = np.round(T, _T_DECIMALS)
    total = viol = 0
    for a in np.unique(asset_id):
        mats = np.sort(np.unique(Tkey[asset_id == a]))
        smiles = {}
        for tv in mats:
            m = (asset_id == a) & (Tkey == tv)
            if m.sum() < 2:
                continue
            kk = k[m]
            order = np.argsort(kk)
            smiles[tv] = (kk[order], total_variance(iv[m][order], tv))
        avail = [tv for tv in mats if tv in smiles]
        for t1, t2 in zip(avail, avail[1:]):
            k1, w1 = smiles[t1]
            k2, w2 = smiles[t2]
            lo = max(k1.min(), k2.min())
            hi = min(k1.max(), k2.max())
            if hi - lo < 1e-9:
                continue
            grid = np.linspace(lo, hi, n_grid)
            w1g = np.interp(grid, k1, w1)
            w2g = np.interp(grid, k2, w2)
            total += grid.size
            viol += int(np.sum(w1g - w2g > 1e-10))
    return 100.0 * viol / total if total else 0.0
