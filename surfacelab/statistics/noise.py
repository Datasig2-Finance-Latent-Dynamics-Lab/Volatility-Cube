"""
Own-observation noise-to-signal ratio r, and how ρ and r move with quote count.

WHY THIS FILE EXISTS
--------------------
The exploitability ceiling is C = ρ·r / (1-ρ + r) (see ceiling.py).  ρ we get from
correlation.py; this file produces the OTHER half: r.

    r = ν² / σ²
      ν² = variance of the error in estimating today's ATM from your own quotes
           (the "prior fit quality / few-quotes" noise — large when fits are poor)
      σ² = variance of the true day-over-day ATM increment (the signal to be predicted)

Intuition for why r is the lever: cross-asset info can only help to the extent your OWN
observation has left you uncertain.  If your own quotes pin today's ATM tightly (small
r), there is nothing for a peer to add, no matter how correlated.  Cross-asset becomes
worth it only when r is large — i.e. when your own fit is bad.  This file measures r
*as a function of the number of own quotes N*, which is exactly the regime sweep the
analysis needs.

NOTE on the quote-count sweep.  In the main validation harness, quote count is fixed
small by design (few for fitting, the rest held out), so it is not a useful stratifier
THERE.  Here it is the whole point: we deliberately vary how many quotes the ATM
estimator sees, to trace how own-noise r (and the *observed* correlation ρ̂) decay as
information accrues.  N=10 is a reasonable reference, but we sweep a grid.
"""
from __future__ import annotations

import numpy as np

from .atm import atm_iv, pairwise_corr


def _rng(seed: int) -> np.random.Generator:
    # Explicit Generator so the sweep is reproducible (no global-state randomness).
    return np.random.default_rng(seed)


def noise_and_estimates(ds, days: np.ndarray, n_quotes: int, n_draws: int = 8,
                        min_full: int = 8, seed: int = 0):
    """For a given own-quote budget N, return per-day estimated ATM and the noise ν².

    For each (day, asset) with enough quotes:
      • ATM_full  = estimator on ALL the asset's quotes that day  (the reference "truth")
      • ATM_N     = estimator on a random N-subset, repeated n_draws times
      • squared error (ATM_N - ATM_full)² accumulated  ->  ν²(N) once averaged

    WHY compare to the full-data fit rather than an external truth: ν must be the error of
    the SAME estimator you actually use, just starved of data.  The full-data fit is the
    best that estimator can do; the gap to the N-quote fit is precisely the own-observation
    noise that cross-asset info would have to beat.

    Also returns one ATM_N draw per (day, asset) as a series, so the caller can measure
    the *observed* cross-asset correlation ρ̂(N) — which is attenuated by this very noise.
    """
    rng = _rng(seed)
    N = ds.n_assets
    est_series = np.full((len(days), N), np.nan)   # one noisy ATM_N draw per day/asset
    full_series = np.full((len(days), N), np.nan)  # ATM_full per day/asset
    sq_err = np.zeros(N)
    cnt = np.zeros(N)
    for i, t in enumerate(days):
        valid = ds.query_feats[t, :, 1] > 0
        for a in range(N):
            m = np.where(valid & (ds.asset_ids[t] == a))[0]
            if len(m) < min_full:
                continue
            lm, iv = ds.query_feats[t, m, 0], ds.targets[t, m]
            atm_full = atm_iv(lm, iv)
            if not np.isfinite(atm_full):
                continue
            full_series[i, a] = atm_full
            for d in range(n_draws):
                k = min(n_quotes, len(m))
                sub = rng.choice(len(m), size=k, replace=False)
                atm_n = atm_iv(lm[sub], iv[sub])
                if np.isfinite(atm_n):
                    sq_err[a] += (atm_n - atm_full) ** 2
                    cnt[a] += 1
                    if d == 0:
                        est_series[i, a] = atm_n   # keep the first draw as the "observed" series
    nu2 = np.where(cnt > 0, sq_err / np.maximum(cnt, 1), np.nan)   # ν²(N) per asset
    return {"nu2": nu2, "est_series": est_series, "full_series": full_series}


def r_for_quotecount(ds, days: np.ndarray, n_quotes: int, sigma2: np.ndarray,
                     **kw) -> dict:
    """Assemble r(N) = ν²(N)/σ² per asset and pooled, plus the observed ρ̂(N).

    σ² (signal variance) is passed in from the full-data increments so it is fixed across
    the N-sweep — only the numerator ν² moves with N.  That isolates the effect of own
    information: r should fall monotonically toward ~0 as N grows.

    ρ̂(N) is the cross-asset correlation of the *N-quote-estimated* increments.  Because
    each ATM_N carries independent measurement noise, the observed correlation is
    attenuated:  ρ̂ ≈ ρ_true / (1 + 2r).  Reporting ρ̂(N) alongside r(N) makes a second
    point concrete — at small N you don't even SEE the full correlation, so a model
    trained on sparse quotes has even less cross-asset signal to latch onto than the
    true ρ suggests.
    """
    out = noise_and_estimates(ds, days, n_quotes, **kw)
    nu2 = out["nu2"]
    r = nu2 / sigma2
    # observed correlation of the noisy N-quote increments
    d_est = np.diff(out["est_series"], axis=0)
    rho_hat = pairwise_corr(d_est)
    r_mean = float(np.nanmean(r)) if np.any(np.isfinite(r)) else np.nan
    return {"n_quotes": n_quotes, "r_per_asset": r, "r_mean": r_mean,
            "nu2": nu2, "rho_hat": rho_hat}
