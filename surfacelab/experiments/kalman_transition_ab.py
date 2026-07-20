"""
A/B the Kalman SSVI transition: parameter LEVELS (z_t = A z_{t-1}) vs INCREMENTS
(Δz_t = A Δz_{t-1}), both keeping the full cross-asset coupling A.

Runs the sequential harness on market data in two regimes:
  • standard      — every asset keeps its own context (run_sequential)
  • exclude AAPL  — AAPL gets NO context, must be rebuilt from peers via A's
                    off-diagonal blocks (run_sequential_exclude) → the cross-asset test

Also saves heatmaps of the learned A (levels) and A (increments) so the SPY-leads
structure can be compared between the two.

    .venv/bin/python -m surfacelab.experiments.kalman_transition_ab
"""
from __future__ import annotations

import warnings

import numpy as np

from surfacelab.experiments.configs import get_experiment
from surfacelab.models.kalman_ssvi import KalmanSSVIModel, _D
from surfacelab.eval import run_sequential, run_sequential_exclude

warnings.filterwarnings("ignore")

OUT = "results/surfacelab/kalman_transition_ab"
PARAM_NAMES = ["v_0", "v_inf", "kappa", "rho", "eta", "gamma"]


def heatmap(A_core, asset_names, title, path):
    """Save a heatmap of the 6N×6N coupling, blocked by asset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(asset_names)
    n = A_core.shape[0]
    v = np.abs(A_core).max()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(A_core, cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="coefficient")
    # asset block gridlines + labels
    for b in range(1, N):
        ax.axhline(b * _D - 0.5, color="k", lw=0.6, alpha=0.4)
        ax.axvline(b * _D - 0.5, color="k", lw=0.6, alpha=0.4)
    ticks = [a * _D + (_D - 1) / 2 for a in range(N)]
    ax.set_xticks(ticks); ax.set_xticklabels(asset_names, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(asset_names)
    ax.set_xlabel("source asset  (yesterday's params)")
    ax.set_ylabel("target asset  (today's params)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def summarise(rec, label):
    """Print RMSE per (model, split) for the kalman models; split = {regime}_{ctx}[_excl]."""
    rows = [r for r in rec.summary() if "kalman_ssvi" in r.get("model", "")]
    print(f"\n── {label} ──")
    print(f"  {'split':<16} {'kalman_ssvi (lvl)':>18} {'kalman_ssvi_inc':>18} {'Δ inc-lvl':>12}")
    by_split: dict = {}
    for r in rows:
        by_split.setdefault(r["split"], {})[r["model"]] = r["rmse"]
    for split in sorted(by_split):
        d = by_split[split]
        lvl = d.get("kalman_ssvi", float("nan"))
        inc = d.get("kalman_ssvi_inc", float("nan"))
        print(f"  {split:<16} {lvl:>18.5f} {inc:>18.5f} {inc - lvl:>+12.5f}")


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    exp = get_experiment("market_all_methods_sequential")
    print(f"Loading market data …")
    dataset, _ = exp.loader()
    asset_names = list(dataset.meta.get("asset_names")
                       or [f"A{i}" for i in range(dataset.n_assets)])
    print(f"  {dataset.n_days} days, {dataset.n_assets} assets")

    lvl = KalmanSSVIModel(transition_mode="levels")
    inc = KalmanSSVIModel(transition_mode="increments")
    for m in (lvl, inc):
        print(f"Training {m.name} …", flush=True)
        m.train(dataset, saved=False, force=True)

    # ── heatmaps of the learned coupling ──
    print("Heatmaps:")
    heatmap(lvl.A_core, asset_names, "Kalman SSVI — coupling A on LEVELS  (z_t = A z_{t-1})",
            f"{OUT}/A_levels.png")
    heatmap(inc.A_core, asset_names, "Kalman SSVI — coupling A on INCREMENTS  (Δz_t = A Δz_{t-1})",
            f"{OUT}/A_increments.png")

    ctx = (10, 50, 200, 500)
    print("\nRunning standard sequential harness (each asset keeps its context) …", flush=True)
    rec = run_sequential([lvl, inc], dataset, ctx_sizes=ctx)
    summarise(rec, "standard sequential (own context)")

    print("\nRunning leave-one-out sequential harness (AAPL has NO context) …", flush=True)
    lvl2 = KalmanSSVIModel(transition_mode="levels")
    inc2 = KalmanSSVIModel(transition_mode="increments")
    for m in (lvl2, inc2):
        m.train(dataset, saved=False, force=True)
    rec_ex = run_sequential_exclude([lvl2, inc2], dataset, "AAPL", ctx_sizes=ctx)
    summarise(rec_ex, "exclude AAPL (cross-asset extrapolation)")

    rec.save(OUT)
    print(f"\nWrote heatmaps + records to {OUT}/")


if __name__ == "__main__":
    main()
