"""
Cross-asset correlation ρ and the common-factor structure.

WHY THIS FILE EXISTS
--------------------
ρ is the headline number people quote to argue "the assets co-move, so cross-asset
info must help."  This file measures it properly and, crucially, decomposes it.  The
key reframing of the whole investigation:

    A high *marginal* correlation does NOT imply exploitable information.  What can be
    exploited is only the part of one asset's increment that (a) is shared with the
    others AND (b) is NOT already a single common factor you can read off any one of
    them (here: SPY).

So we report three things:
  • the raw correlation matrix and mean (what you see at first glance),
  • the partial correlation given SPY removed (what survives a single market factor),
  • the common-factor variance share via PCA (the hard CAP on cross-asset usefulness:
    no amount of other-asset data can ever explain the idiosyncratic remainder).
"""
from __future__ import annotations

import numpy as np


def mean_offdiag(C: np.ndarray, idx: list[int]) -> float:
    """Mean of the off-diagonal correlations among the assets in `idx`."""
    vals = [C[i, j] for k, i in enumerate(idx) for j in idx[k + 1:]]
    return float(np.mean(vals)) if vals else np.nan


def partial_given(C: np.ndarray, ctrl: int) -> np.ndarray:
    """Partial-correlation matrix controlling for one asset (e.g. SPY).

    partial(i,j | c) = (r_ij - r_ic r_jc) / sqrt((1-r_ic^2)(1-r_jc^2))

    WHY: if the assets co-move ONLY through a single market factor, then conditioning on
    that factor (SPY) should collapse the residual stock-stock correlation toward 0.
    Whatever partial correlation REMAINS is the genuinely multi-factor coupling — the
    only thing a graph/cross-asset model could add beyond a one-factor baseline.  If it
    is ~0, cross-asset edges are redundant; if it is sizeable (it is here, ≈0.27), the
    question becomes whether that residual is *large enough relative to own noise* to
    matter — which is what ceiling.py answers.
    """
    N = C.shape[0]
    P = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            denom = np.sqrt((1 - C[i, ctrl] ** 2) * (1 - C[j, ctrl] ** 2))
            P[i, j] = P[j, i] = (C[i, j] - C[i, ctrl] * C[j, ctrl]) / denom if denom > 0 else np.nan
    return P


def factor_variance_share(C: np.ndarray) -> dict:
    """How much of the cross-section is one common factor, via eigen-decomposition of C.

    For an equicorrelation model with correlation ρ̄, the top eigenvalue is 1+(N-1)ρ̄ and
    its share of the trace is [1+(N-1)ρ̄]/N → ρ̄ as N grows.  So:
      • top_pc_share  ≈ the fraction of increment variance that is COMMON (factor-driven)
      • 1 - top_pc_share ≈ the IDIOSYNCRATIC fraction that cross-asset info can NEVER touch

    WHY this is the ceiling's backbone: even with infinitely many, perfectly observed
    peers you can at best learn the common factor exactly; the idiosyncratic remainder
    is irreducible.  So this single number upper-bounds the relevance of every method
    you have tried.
    """
    w = np.sort(np.linalg.eigvalsh(C))[::-1]
    N = C.shape[0]
    return {
        "eigenvalues": w,
        "top_pc_share": float(w[0] / N),          # common-factor variance share
        "idiosyncratic_share": float(1 - w[0] / N),
        "n_factors_for_90pct": int(np.searchsorted(np.cumsum(w) / N, 0.90) + 1),
    }


def summarise(C: np.ndarray, asset_names: list[str], spy_name: str = "SPY") -> dict:
    """Bundle the correlation story into one dict for reporting."""
    spy = asset_names.index(spy_name)
    stocks = [i for i in range(len(asset_names)) if i != spy]
    P = partial_given(C, spy)
    fac = factor_variance_share(C)
    return {
        "mean_stock_stock_raw": mean_offdiag(C, stocks),
        "mean_stock_spy": float(np.mean([C[i, spy] for i in stocks])),
        "mean_stock_stock_partial_given_spy": mean_offdiag(P, stocks),
        "common_factor_share": fac["top_pc_share"],
        "idiosyncratic_share": fac["idiosyncratic_share"],
        "n_factors_for_90pct": fac["n_factors_for_90pct"],
    }
