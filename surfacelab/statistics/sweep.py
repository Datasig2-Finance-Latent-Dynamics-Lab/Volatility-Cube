"""
The quote-count sweep: how ρ̂ and r (and hence the ceiling C) move with own quotes.

WHY THIS FILE EXISTS
--------------------
This answers the concrete request: "plot how ρ and r change as we increase the number of
quotes."  For each quote budget N we compute:
  • r(N)      — own-observation noise-to-signal (falls as N grows: more quotes, better fit)
  • ρ̂(N)      — the OBSERVED cross-asset correlation of the N-quote increments
                (rises toward the true ρ as N grows: less measurement attenuation)
  • C(N)      — the exploitability ceiling using the TRUE ρ but the N-dependent r
  • C_obs(N)  — the ceiling a model could actually chase, using ρ̂(N) (attenuated) and r(N)

The two ceilings bracket the story: even with the true ρ, C is small unless r is large;
and at small N the *observable* ρ̂ is smaller still, so C_obs is smaller again.  N=10 is
the reference the rest of the project uses, but the sweep shows the whole trajectory.
"""
from __future__ import annotations

import numpy as np

from .atm import atm_series, increments, pairwise_corr
from .correlation import mean_offdiag
from .ceiling import ceiling, realizable_gain
from .noise import r_for_quotecount


def run_sweep(ds, n_grid=(3, 5, 8, 10, 15, 20, 30, 50, 100), seed: int = 0) -> dict:
    """Full sweep over quote budgets; returns true ρ/σ² and a per-N table."""
    days = ds.train_idx()
    names = ds.meta["asset_names"]
    spy = names.index("SPY") if "SPY" in names else None
    stocks = [i for i in range(ds.n_assets) if i != spy]

    # --- truth from full-data fits: σ² (signal) and true ρ (common share) ---
    atm = atm_series(ds, days)
    d_full = increments(atm)
    sigma2 = np.nanvar(d_full, axis=0)                 # per-asset increment variance (the signal)
    C_true = pairwise_corr(d_full)
    rho_true = mean_offdiag(C_true, stocks)            # mean stock-stock correlation

    rows = []
    for N in n_grid:
        res = r_for_quotecount(ds, days, N, sigma2, seed=seed)
        r_mean = res["r_mean"]
        rho_hat = mean_offdiag(res["rho_hat"], stocks)
        n_peers = ds.n_assets - 1
        rows.append({
            "n_quotes": N,
            "r_mean": r_mean,
            "rho_hat": rho_hat,                        # observed (attenuated) correlation
            "ceiling_true_rho": ceiling(rho_true, r_mean),     # ORACLE bound (perfect peers)
            # HONEST realizable gain: peers share your noise (symmetric), but ρ assumed known
            "realizable_sym": realizable_gain(rho_true, r_mean, n_peers),
            "nu2": res["nu2"],                         # per-asset noise (for asymmetric.py)
        })
    return {"rho_true": rho_true, "sigma2": sigma2, "rows": rows,
            "asset_names": names, "corr_full": C_true}
