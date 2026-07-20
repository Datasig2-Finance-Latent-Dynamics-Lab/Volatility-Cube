"""
Diagnostic: should the Kalman SSVI transition act on parameter *levels* or *increments*?

Builds the fitted-SSVI parameter time series Z (n_days, N*6) the same way
`KalmanSSVIModel.train` does, then asks two questions:

  1. Do daily increments Δθ have any lag-1 autocorrelation? (If ~0, an increment-AR
     momentum term collapses to a plain random walk.)
  2. Which one-step transition predicts held-out days best:
        RW    : ẑ_t = z_{t-1}                 (Δ = 0, random walk)
        LevAR : ẑ_t = A_lev z_{t-1}           (current model — mean-reverting AR(1))
        IncAR : ẑ_t = z_{t-1} + A_inc Δz_{t-1} (the proposed increment-AR(1))

All errors are reported in per-dimension standardised units so the six SSVI params
(v0~0.04 vs η~1) are comparable, and as a skill score relative to RW.

    .venv/bin/python -m surfacelab.experiments.ar_diagnostic --config heston_all_methods
    .venv/bin/python -m surfacelab.experiments.ar_diagnostic --config market_all_methods
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from surfacelab.experiments.configs import get_experiment
from surfacelab.models.kalman import fit_transition
from surfacelab.models.kalman_ssvi import _ffill_bfill, _D
from surfacelab.models.parametric.representations import fit_ssvi_fast

warnings.filterwarnings("ignore")

PARAM_NAMES = ["v_0", "v_inf", "kappa", "rho", "eta", "gamma"]


def build_Z(dataset, n_history=250, maxiter=150) -> np.ndarray:
    """Per-(asset, day) warm-started SSVI fits → stacked state series (mirrors train())."""
    N = int(dataset.n_assets)
    idx = dataset.train_idx()
    if n_history and len(idx) > n_history:
        idx = idx[-n_history:]
    Z = np.full((len(idx), N * _D), np.nan)
    last: dict[int, np.ndarray | None] = {a: None for a in range(N)}
    for i, t in enumerate(idx):
        q = dataset.quotes_at(t)
        for a in range(N):
            sel = q.asset_id == a
            if sel.sum() >= _D:
                st = fit_ssvi_fast(q.k[sel], q.T[sel], q.iv[sel],
                                   x0=last[a], maxiter=maxiter)
                p = np.array([st.v_0, st.v_inf, st.kappa, st.rho, st.eta, st.gamma])
                last[a] = p
                Z[i, a * _D:(a + 1) * _D] = p
    return _ffill_bfill(Z)


def lag1_autocorr(DZ: np.ndarray) -> np.ndarray:
    """Per-dimension lag-1 autocorrelation of the increment series."""
    out = np.zeros(DZ.shape[1])
    for j in range(DZ.shape[1]):
        x = DZ[:, j] - DZ[:, j].mean()
        denom = (x * x).sum()
        out[j] = (x[1:] * x[:-1]).sum() / denom if denom > 1e-12 else 0.0
    return out


def onestep_eval(Z: np.ndarray, frac_train=0.7):
    """Fit each transition on the first `frac_train` of Z, score one-step MSE on the tail."""
    n = Z.shape[0]
    cut = int(n * frac_train)
    Ztr = Z[:cut]
    DZtr = np.diff(Ztr, axis=0)

    A_lev, _ = fit_transition(Ztr, transition_type="full")
    A_inc, _ = fit_transition(DZtr, transition_type="full")
    A_inc_d, _ = fit_transition(DZtr, transition_type="diagonal")  # per-param scalar AR

    # standardise errors per dimension by the train-set increment scale
    scale = np.where(DZtr.std(axis=0) < 1e-8, 1.0, DZtr.std(axis=0))

    se = {"RW": [], "LevAR": [], "IncAR": [], "IncARd": []}
    for t in range(cut, n):
        z_prev = Z[t - 1]
        d_prev = Z[t - 1] - Z[t - 2]
        truth = Z[t]
        preds = {
            "RW": z_prev,
            "LevAR": A_lev @ z_prev,
            "IncAR": z_prev + A_inc @ d_prev,
            "IncARd": z_prev + A_inc_d @ d_prev,
        }
        for k, p in preds.items():
            se[k].append(((p - truth) / scale) ** 2)
    mse = {k: np.mean(v, axis=0) for k, v in se.items()}  # per-dim, standardised
    return mse, A_lev, A_inc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="heston_all_methods")
    ap.add_argument("--frac-train", type=float, default=0.7)
    args = ap.parse_args()

    exp = get_experiment(args.config)
    print(f"Loading '{exp.name}' …")
    dataset, _ = exp.loader()
    N = int(dataset.n_assets)
    print(f"  {dataset.n_days} days, {N} assets — fitting SSVI per (asset, day) …")
    Z = build_Z(dataset)
    DZ = np.diff(Z, axis=0)
    print(f"  Z: {Z.shape}  ({DZ.shape[0]} increments)\n")

    # ── Q1: increment autocorrelation, averaged across assets per SSVI param ──
    ac = lag1_autocorr(DZ)                      # (N*6,)
    ac_by_param = ac.reshape(N, _D).mean(axis=0)
    band = 2.0 / np.sqrt(DZ.shape[0])           # rough ±2/√n white-noise band
    print("Q1  lag-1 autocorrelation of increments  (|ρ| > %.3f ≈ significant)" % band)
    print("    param      mean ρ(Δθ_t, Δθ_{t-1})   verdict")
    for j, name in enumerate(PARAM_NAMES):
        flag = "momentum" if abs(ac_by_param[j]) > band else "~white"
        print(f"    {name:<8}  {ac_by_param[j]:+8.3f}              {flag}")

    # ── Q2: out-of-sample one-step prediction ──
    mse, A_lev, A_inc = onestep_eval(Z, args.frac_train)
    by = {k: v.reshape(N, _D).mean(axis=0) for k, v in mse.items()}  # avg across assets
    print("\nQ2  out-of-sample one-step MSE (standardised; lower = better)")
    print("    param        RW      LevAR     IncAR    IncARd  | IncARd skill vs RW")
    for j, name in enumerate(PARAM_NAMES):
        skill = 1.0 - by["IncARd"][j] / max(by["RW"][j], 1e-12)
        print(f"    {name:<8} {by['RW'][j]:8.4f} {by['LevAR'][j]:8.4f} "
              f"{by['IncAR'][j]:8.4f} {by['IncARd'][j]:8.4f}  | {skill:+6.1%}")
    tot = {k: float(np.mean(v)) for k, v in mse.items()}
    print(f"\n    aggregate   RW={tot['RW']:.4f}  LevAR={tot['LevAR']:.4f}  "
          f"IncAR(full)={tot['IncAR']:.4f}  IncAR(diag)={tot['IncARd']:.4f}")
    print(f"    IncAR full vs RW : {1 - tot['IncAR']/tot['RW']:+.1%} skill")
    print(f"    IncAR diag vs RW : {1 - tot['IncARd']/tot['RW']:+.1%} skill")
    print(f"    LevAR     vs RW : {1 - tot['LevAR']/tot['RW']:+.1%} skill")

    # how close is the level-AR diagonal to 1 (i.e. is it just approximating a RW)?
    diag = np.clip(np.diag(A_lev), -2, 2).reshape(N, _D).mean(axis=0)
    print("\n    mean diag(A_lev) per param (≈1 ⇒ LevAR is straining to be a random walk):")
    print("    " + "  ".join(f"{n}={d:.2f}" for n, d in zip(PARAM_NAMES, diag)))

    print("\nRead-off:")
    print("  • IncAR skill ≤ 0 and Q1 ~white  → momentum term is noise; use a RANDOM WALK.")
    print("  • IncAR clearly > RW and > LevAR → the increment-AR(1) term is justified; build it.")
    print("  • LevAR worse than RW            → confirms the zero-mean reversion is hurting.")


if __name__ == "__main__":
    main()
