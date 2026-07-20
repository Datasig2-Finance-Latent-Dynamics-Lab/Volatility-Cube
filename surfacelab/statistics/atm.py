"""
ATM level extraction and day-over-day increments.

WHY THIS FILE EXISTS
--------------------
The whole "is cross-asset coupling exploitable?" question is about ONE scalar per
asset per day: the at-the-money (ATM) implied-vol *level*, and how it moves day to
day (the *increment* Δ ATM).  Everything downstream — the cross-asset correlation ρ,
the own-observation noise r, the exploitability ceiling — is built on this series.

We define ATM with the SAME estimator the project already uses to build its B-spline
prior (`surfacelab.data.prior._fit_smile`), fitted to a day/asset's quotes pooled in
log-moneyness and evaluated at lm = 0.  Using the project's own fitter matters: the
"prior fit quality" we later call r is literally the sampling error of THIS estimator,
not of some idealised one.  (Empirically this pooled estimator reproduces the existing
results/diagnostics/cross_asset_corr.txt numbers: mean stock-stock ρ ≈ 0.42, so it is
the same definition that produced that diagnostic.)
"""
from __future__ import annotations

import numpy as np

from surfacelab.data.prior import _fit_smile


def atm_iv(lm: np.ndarray, iv: np.ndarray, degree: int = 3) -> float:
    """ATM implied vol from one asset/day's quotes: B-spline smile in lm, read at lm=0.

    WHY pooled across maturities: near the money the smile is flat enough that a single
    lm-smile gives a stable ATM read even from a handful of quotes, whereas a strict
    per-maturity interpolation often has no point bracketing lm=0 when quotes are sparse
    (exactly the few-quote regime we care about).  A consistent estimator at every quote
    count is what lets us measure how its *error* shrinks as quotes increase.
    """
    if len(lm) < 2:
        return np.nan
    # _fit_smile already falls back to a flat mean when there are too few points for a
    # spline, so very small quote budgets still return a (cruder) ATM rather than NaN —
    # essential for the low-N end of the sweep, which is exactly the regime of interest.
    f = _fit_smile(np.asarray(lm, float), np.asarray(iv, float), degree)
    return float(f(np.array([0.0]))[0])


def atm_series(ds, days: np.ndarray | None = None, min_quotes: int = 4) -> np.ndarray:
    """ATM-IV matrix (n_days, n_assets) over `days` (default: training days).

    The target object of the whole analysis.  NaN where an asset has < min_quotes that
    day (the increment code handles NaNs pairwise so one missing asset never discards a
    whole day across the board).
    """
    days = ds.train_idx() if days is None else np.asarray(days)
    N = ds.n_assets
    out = np.full((len(days), N), np.nan)
    for i, t in enumerate(days):
        valid = ds.query_feats[t, :, 1] > 0           # T>0 marks real (non-padded) points
        for a in range(N):
            m = valid & (ds.asset_ids[t] == a)
            if m.sum() >= min_quotes:
                out[i, a] = atm_iv(ds.query_feats[t, m, 0], ds.targets[t, m])
    return out


def increments(atm: np.ndarray) -> np.ndarray:
    """Day-over-day Δ ATM, (n_days-1, n_assets).

    WHY increments, not levels: ATM levels are highly persistent (near random walks), so
    a level correlation is dominated by shared trend and overstates exploitable signal.
    The increment is the genuinely-uncertain quantity a model must predict each day, and
    it is the increment whose cross-asset correlation could in principle be exploited.
    """
    return np.diff(atm, axis=0)


def pairwise_corr(x: np.ndarray) -> np.ndarray:
    """Correlation matrix using pairwise-complete observations (NaN-robust).

    np.corrcoef drops any row with a single NaN, which here would throw away ~20% of
    days because some thin name is missing.  Pairwise-complete keeps every usable pair,
    matching how the existing diagnostic reached n≈956 usable increments.
    """
    N = x.shape[1]
    C = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            g = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
            if g.sum() > 2:
                C[i, j] = C[j, i] = np.corrcoef(x[g, i], x[g, j])[0, 1]
    return C
