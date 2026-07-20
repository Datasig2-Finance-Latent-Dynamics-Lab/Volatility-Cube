"""
Asymmetric liquidity: what if SPY (or every peer) is well-observed but the TARGET is sparse?

WHY THIS FILE EXISTS
--------------------
The symmetric analysis (sweep.py / ceiling.py) shows that when EVERY asset is equally
data-starved, cross-asset info is useless — peers are as blind as you are.  But that is
an artefact of how the evaluation harness samples context: `eval.harness._per_asset_sample`
draws the SAME n_ctx quotes for every asset, SPY included.  The real market is the
opposite: SPY and the big names are extremely liquid, while a given single name can be
sparse.  THAT asymmetry is precisely the regime where a correlated, well-observed peer
should let you pin a sparse asset's move — and it is the regime the symmetric harness
hides.

This file computes, on the EMPIRICAL increment covariance (no equicorrelation idealisation),
how much a target asset's residual ATM-increment variance drops when some peers are
observed exactly (e.g. "perfect SPY", or "everything perfect except this one asset") while
the target sees only N noisy quotes.  It is the experiment to run to see whether cross-asset
is exploitable once liquidity is asymmetric.
"""
from __future__ import annotations

import numpy as np


def empirical_cov(d_inc: np.ndarray) -> np.ndarray:
    """Increment covariance Σ built pairwise (NaN-robust): Σ_ij = corr_ij · σ_i · σ_j.

    WHY pairwise + reconstruct rather than np.cov: complete-case cov would discard ~20% of
    days where any thin name is missing.  We take per-asset σ from all available days and
    correlations from pairwise-complete pairs, then a tiny ridge guarantees the matrix is
    invertible for the conditioning below.  Using the EMPIRICAL Σ (not a one-factor model)
    means each asset keeps its real beta to SPY and its real idiosyncratic share.
    """
    N = d_inc.shape[1]
    sd = np.nanstd(d_inc, axis=0)
    C = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            g = ~np.isnan(d_inc[:, i]) & ~np.isnan(d_inc[:, j])
            if g.sum() > 2:
                C[i, j] = C[j, i] = np.corrcoef(d_inc[g, i], d_inc[g, j])[0, 1]
    Sig = C * np.outer(sd, sd)
    Sig += 1e-12 * np.eye(N)            # ridge for numerical PD-ness
    return Sig


def posterior_var(Sigma: np.ndarray, noise: np.ndarray, target: int) -> float:
    """Var(Δθ_target | all noisy observations), exact Gaussian conditioning.

    `noise` is a per-asset observation-noise variance: 0 = observed perfectly, np.inf =
    not observed at all, finite = observed with that noise.  This single primitive expresses
    every scenario — own-only, perfect-SPY, perfect-all-peers, symmetric — just by setting
    the noise vector.  Standard result: with observations y = Δθ + e, e ~ diag(noise),

        Var(target | y_O) = Σ_tt − Σ_tO (Σ_OO + diag(noise_O))⁻¹ Σ_Ot

    over the observed set O (the finite-noise entries).
    """
    O = np.where(np.isfinite(noise))[0]
    M = Sigma[np.ix_(O, O)] + np.diag(noise[O])
    s_tO = Sigma[target, O]
    return float(Sigma[target, target] - s_tO @ np.linalg.solve(M, s_tO))


def asymmetric_gains(Sigma: np.ndarray, nu2: np.ndarray, asset_names: list[str],
                     spy_name: str = "SPY") -> list[dict]:
    """Per-target variance-reduction from peers, under realistic liquidity asymmetries.

    For each target asset (observed with its own measured N-quote noise nu2[target]) we
    compare three conditioning sets against the own-only baseline:
      • perfect_spy   : SPY observed exactly, everything else unobserved
      • perfect_peers : every OTHER asset observed exactly (the upper bound for "all the
                        liquid context you could ever have")
      • pure_spy_R2   : SPY's explanatory power with NO own info (nu2_target → ∞) — i.e.
                        how much of the target's move SPY alone pins down a priori (= beta R²)

    WHY these three: perfect_spy is the cheap, realistic intervention (you almost always
    have a liquid SPY); perfect_peers is the ceiling of cross-asset help; pure_spy_R2
    isolates how much is the market factor alone vs your own quotes.  If perfect_spy is
    large for the sparse names, cross-asset IS exploitable once liquidity is asymmetric —
    the opposite of the symmetric-harness conclusion.
    """
    N = len(asset_names)
    spy = asset_names.index(spy_name)
    rows = []
    for i in range(N):
        if i == spy:
            continue
        inf = np.full(N, np.inf)
        own = inf.copy(); own[i] = nu2[i]
        v_own = posterior_var(Sigma, own, i)

        n_spy = own.copy(); n_spy[spy] = 0.0
        n_all = own.copy(); n_all[[j for j in range(N) if j != i]] = 0.0
        n_pure = inf.copy(); n_pure[spy] = 0.0     # SPY perfect, target unobserved

        rows.append({
            "asset": asset_names[i],
            "own_r": nu2[i] / Sigma[i, i],
            "gain_perfect_spy": 1 - posterior_var(Sigma, n_spy, i) / v_own,
            "gain_perfect_peers": 1 - posterior_var(Sigma, n_all, i) / v_own,
            "pure_spy_R2": 1 - posterior_var(Sigma, n_pure, i) / Sigma[i, i],
        })
    return rows
