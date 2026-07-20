"""
The exploitability ceiling: turning "cross-asset doesn't help" into a number.

WHY THIS FILE EXISTS
--------------------
This is the punchline of the whole module.  Under a one-factor Gaussian model of ATM
increments — common factor share ρ, idiosyncratic share 1-ρ — with your own observation
carrying noise-to-signal r, the MAXIMUM fraction of your residual variance that ANY
cross-asset information can remove is

        C = ρ · r / ( (1-ρ) + r )          (derivation in the module README)

and C is capped at ρ.  This is an *oracle* bound: it assumes the other assets are
observed perfectly and that there are infinitely many of them, so the common factor is
known exactly.  Real peers are themselves estimated from few noisy quotes, so the
achievable gain is strictly below C.  If even this generous ceiling is small in your
regimes, then no method can help, and a CNP that ignores cross-asset edges is making the
*correct* inference, not failing.

Reading C:
  • r → 0  (own quotes pin the ATM):           C → 0      — nothing to add, you already know it
  • r → ∞  (own quotes useless):               C → ρ      — peers can recover the common part, no more
  • the idiosyncratic share (1-ρ) is a hard floor cross-asset can never cross.
"""
from __future__ import annotations

import numpy as np


def ceiling(rho: float, r: float) -> float:
    """C = ρr / ((1-ρ) + r): max fractional reduction in residual VARIANCE from cross-asset.

    WHY this exact form: posterior precision is additive.  Own info contributes precision
    1/ν²; knowing the common factor exactly removes the common share leaving only the
    idiosyncratic σ²(1-ρ).  Combining the two and taking the fractional drop in variance
    gives this closed form (see README for the two-line algebra).  It is the cleanest
    possible statement of "own info vs. shared info".
    """
    return rho * r / ((1.0 - rho) + r)


def realizable_gain(rho: float, r: float, n_peers: int, r_peer: float | None = None) -> float:
    """The HONEST exploitable fraction: peers carry the SAME observation noise you do.

    WHY this matters more than the oracle ceiling C.  C assumes peers reveal the common
    factor perfectly.  But in this dataset every asset has ~the same liquidity, so when
    your own quotes are too sparse to pin your ATM (large r), the peers' quotes are
    equally sparse — they cannot reveal the factor any better than you can.  This is the
    binding constraint the oracle bound hides.

    Exact one-factor, symmetric-noise computation (σ²=1 units).  Predictors are your own
    obs y0 (noise r) and n_peers peer obs (noise r_peer); all share the common factor.
    Best-linear-predictor residual variance via the closed-form Gaussian solve:
        Var(x0 | preds) = 1 - cᵀ S⁻¹ c,   S = predictor cov, c = Cov(x0, preds)
    Returned: fractional variance reduction RELATIVE to using own obs alone.

    Caveat the number still flatters reality: it assumes the coupling ρ is KNOWN.  A
    learned model must instead estimate ρ from data, where the observable ρ̂(N) ≈ ρ/(1+2r)
    is ~0 at the operating quote count — so even this gain is not actually learnable.
    """
    r_peer = r if r_peer is None else r_peer
    m = n_peers + 1
    S = np.full((m, m), rho)
    S[0, 1:] = S[1:, 0] = rho               # own↔peer common-factor covariance
    S[0, 0] = 1.0 + r                       # Var(own obs)
    for j in range(1, m):
        S[j, j] = 1.0 + r_peer              # Var(peer obs)
    c = np.full(m, rho)
    c[0] = 1.0                              # Cov(x0, own obs)=Var(x0)=1
    v_all = 1.0 - c @ np.linalg.solve(S, c)
    v_own = r / (1.0 + r)
    return float((v_own - v_all) / v_own) if v_own > 0 else 0.0


def rmse_gain(C: float) -> float:
    """Translate a variance-reduction ceiling C into an RMSE-reduction ceiling.

    Models are scored on RMSE, not variance, so this is the number to compare against the
    estimation noise of your training loop.  RMSE ∝ sqrt(variance) ⇒ best-case relative
    RMSE improvement = 1 - sqrt(1 - C) ≈ C/2 for small C.  A few-percent C is a sub-percent
    RMSE prize — typically below the run-to-run noise of training, which is exactly why a
    gradient-based learner (the CNP) never bothers to capture it.
    """
    return 1.0 - np.sqrt(max(0.0, 1.0 - C))


def conditional_mi(C: float) -> float:
    """Ceiling on the conditional mutual information I(x_i ; peers | own obs), in bits.

    For jointly-Gaussian variables, MI = -0.5 log2(1 - fractional variance reduction).
    This expresses the SAME ceiling in information units: how many bits about today's ATM
    the peers could add on top of your own quotes.  When this is a small fraction of a
    bit, "the observations already contain too much information" is literally true.
    """
    return -0.5 * np.log2(max(1e-12, 1.0 - C))


def required_r(rho: float, target_C: float) -> float:
    """Invert the ceiling: how bad must own info (r) be for cross-asset to buy target_C?

    WHY useful: lets you state the result as a threshold the data rarely meets, e.g.
    "with ρ=0.42, removing even 10% of residual variance needs r>0.2 — own quotes
    explaining <83% of the variance — a corner the empirical r(N) almost never reaches."
    Solving C = ρr/((1-ρ)+r) for r:  r = (1-ρ)·C / (ρ - C).  Returns inf if target_C ≥ ρ
    (unreachable: you cannot beat the common-factor cap).
    """
    if target_C >= rho:
        return np.inf
    return (1.0 - rho) * target_C / (rho - target_C)


def partial_r2_cross(d_inc: np.ndarray, target: int, own_proxy: np.ndarray | None = None):
    """Model-free check: incremental R² from peers after controlling for an own proxy.

    The Gaussian ceiling assumes a factor structure; this is the assumption-light cousin.
    We regress the target asset's increment on (optionally) an own-information proxy, take
    the residual, then regress that residual on the OTHER assets' increments.  The R² of
    that second regression is the fraction of leftover target variance that peers explain
    — the empirical analogue of C.  If it tracks the closed-form C, the factor model is
    not hiding anything; if it is ~0, peers add nothing beyond own info, full stop.

    own_proxy: a (n,) array standing in for "what your own quotes already told you"
    (e.g. the noisy N-quote ATM increment of the target).  If None, no own control is
    applied and you measure the raw cross-asset explanatory power.
    """
    y = d_inc[:, target]
    peers = np.delete(d_inc, target, axis=1)
    g = ~np.isnan(y) & ~np.isnan(peers).any(axis=1)
    if own_proxy is not None:
        g &= ~np.isnan(own_proxy)
    y, peers = y[g], peers[g]
    if g.sum() < peers.shape[1] + 3:
        return np.nan
    if own_proxy is not None:
        x = own_proxy[g]
        beta = np.polyfit(x, y, 1)
        y = y - np.polyval(beta, x)          # residual after own information
    # OLS of (residual) target on peers; R² = explained / total
    P = np.column_stack([peers, np.ones(len(peers))])
    coef, *_ = np.linalg.lstsq(P, y, rcond=None)
    resid = y - P @ coef
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan
