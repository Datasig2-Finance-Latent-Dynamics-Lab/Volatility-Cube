"""
Monte-Carlo validation of the ceiling formula.

WHY THIS FILE EXISTS
--------------------
The entire argument rests on C = ρr/((1-ρ)+r).  Before trusting it on real data we
confirm, by simulation of the one-factor model it assumes, that:

  1. the closed-form C matches the variance reduction an ORACLE (common factor known
     exactly) actually achieves — i.e. the algebra is right; and
  2. the REALISTIC reduction, using a finite number of peers each observed with their
     own noise, sits strictly BELOW C — i.e. C is genuinely an upper bound and the real
     prize is smaller still.

This is the cheap, CNP-free sanity check that licenses every empirical number the rest
of the module reports.  (The expensive complement — proving the CNP *could* exploit
cross-asset if it were there, via a high-r positive control — is described in README.)
"""
from __future__ import annotations

import numpy as np

from .ceiling import ceiling


def simulate(rho: float, r: float, n_peers: int, r_peer: float | None = None,
             n: int = 100_000, seed: int = 0) -> dict:
    """One-factor MC: x_i = sqrt(ρ) F + sqrt(1-ρ) z_i, own obs y_i = x_i + noise(var r).

    Compares three residual variances of the target x_0 (all unit total variance):
      • V_own   : posterior var given only own obs               -> r/(1+r)
      • V_oracle: posterior var given own obs + F known exactly   -> the ceiling case
      • V_real  : posterior var given own obs + n_peers noisy peers (each noise var r_peer)

    Returns realised fractional reductions vs. the closed-form ceiling so the caller can
    print "formula vs simulation" side by side.
    """
    rng = np.random.default_rng(seed)
    r = max(float(r), 1e-4)                                  # guard degenerate r=0/NaN
    r_peer = r if r_peer is None else max(float(r_peer), 1e-4)
    F = rng.standard_normal(n)
    z = rng.standard_normal((n, 1 + n_peers))
    x = np.sqrt(rho) * F[:, None] + np.sqrt(1 - rho) * z      # latent increments
    # noisy own + peer observations
    y_own = x[:, 0] + np.sqrt(r) * rng.standard_normal(n)
    y_peer = x[:, 1:] + np.sqrt(r_peer) * rng.standard_normal((n, n_peers))

    def resid_var(predictors):
        """Var of x_0 after best linear prediction from the given columns.

        Computed from the sample covariance (Var(y) - Σ_yP Σ_PP⁻¹ Σ_Py) rather than a
        least-squares solve: for n≈1e5 rows with near-collinear predictors the explicit
        normal-equations form via np.linalg.solve is far more stable than lstsq's SVD.
        """
        M = np.column_stack([x[:, 0]] + predictors)
        Cov = np.cov(M, rowvar=False)
        s_yy, s_yP, S_PP = Cov[0, 0], Cov[0, 1:], Cov[1:, 1:]
        return float(s_yy - s_yP @ np.linalg.solve(S_PP, s_yP))

    v_own = resid_var([y_own])
    v_oracle = resid_var([y_own, F])                 # F known exactly = infinite perfect peers
    v_real = resid_var([y_own] + [y_peer[:, j] for j in range(n_peers)])
    base = np.var(x[:, 0])
    return {
        "rho": rho, "r": r, "n_peers": n_peers, "r_peer": r_peer,
        "C_formula": ceiling(rho, r),
        "C_oracle_sim": (v_own - v_oracle) / v_own,      # should match C_formula
        "C_realistic_sim": (v_own - v_real) / v_own,     # finite noisy peers; < C_formula
        "var_total": base,
    }


def validation_table(rho: float, r_grid, n_peers: int = 7, **kw) -> list[dict]:
    """Run `simulate` across an r-grid; rows are ready to print/save.

    Reading the output: C_formula and C_oracle_sim should agree to MC error (validates
    the algebra); C_realistic_sim < C_formula always (confirms the ceiling really is a
    ceiling, and shows how much the finite/noisy peers fall short of it).
    """
    return [simulate(rho, float(r), n_peers, **kw) for r in r_grid]
