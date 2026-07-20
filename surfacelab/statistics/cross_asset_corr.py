"""
Cross-asset comovement diagnostic: is the cross-asset signal a single ATM level factor,
with point-by-point WING coupling essentially spurious?

Two correlation matrices over a (log-moneyness, maturity) grid, computed on DAILY IV
INCREMENTS (the quantity the Kalman / graph couplings actually act on):

  1. lead -> surface : corr( SPY's ATM increment ,  TARGET's increment at every grid point )
     How well one peer ATM move explains the target across its whole surface.  Stays high
     even in the wings because a common market level/vega factor lifts the whole surface.

  2. point-by-point  : corr( TARGET increment ,  PEER increment )  at the SAME (k, T)
     Strong at the money, ~0 in the wings: each asset's wing-specific moves are idiosyncratic.

The gap between (1) and (2) in the wings is the result: the genuine cross-asset structure is
a rank-1 level factor; a dense point-wise coupling (e.g. a full Kalman transition over the
SSVI/coefficient increments) fits wing noise and, once it observes peer wings precisely,
actively harms the target — which is exactly the unif-vs-extrap degradation seen in the
leave-one-asset-out experiments.

Run:
    .venv/bin/python -m surfacelab.statistics.cross_asset_corr
Writes the heatmap to results/surfacelab/diagnostics/<peer>_<target>_xcorr.png and prints
the ATM-vs-wing correlations.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

# allow direct `python surfacelab/statistics/cross_asset_corr.py` as well as `-m`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surfacelab.experiments.configs import _market_thesis, REPO
from surfacelab.models.factors import interp_linear_nearest

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
TARGET = "AAPL"           # surface we correlate
PEER = "SPY"              # the lead / reference asset
ATM_K, ATM_T = 0.0, 0.3   # the peer's ATM reference point for matrix (1)
KS = np.linspace(-0.4, 0.4, 17)
TS = np.array([0.10, 0.25, 0.50, 0.75, 1.00])
OUT = REPO / "results" / "surfacelab" / "diagnostics" / f"{PEER}_{TARGET}_xcorr.png"


def _surface_series(ds, asset_id: int, gpts: np.ndarray) -> np.ndarray:
    """Interpolate each day's scattered surface for one asset onto the fixed grid →
    (n_days, n_grid); a day with too few quotes is NaN."""
    out = []
    for t in range(ds.n_days):
        v = ds.valid_points(t)
        idx = v[ds.asset_ids[t, v] == asset_id]
        if idx.size >= 5:
            out.append(interp_linear_nearest(ds.query_feats[t, idx], ds.targets[t, idx], gpts))
        else:
            out.append(np.full(len(gpts), np.nan))
    return np.asarray(out)


def _corr_to_vector(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation of x:(n,) against every column of Y:(n,G) → (G,)."""
    x = x - x.mean()
    Y = Y - Y.mean(0)
    return (x @ Y) / (np.sqrt((x * x).sum()) * np.sqrt((Y * Y).sum(0)) + 1e-12)


def main() -> None:
    ds, _ = _market_thesis()()
    names = list(ds.meta["asset_names"])
    a_tgt, a_peer = names.index(TARGET), names.index(PEER)

    KK, TT = np.meshgrid(KS, TS, indexing="ij")
    gpts = np.stack([KK.ravel(), TT.ravel()], axis=1)

    dT = np.diff(_surface_series(ds, a_tgt, gpts), axis=0)     # daily IV increments
    dP = np.diff(_surface_series(ds, a_peer, gpts), axis=0)
    ok = np.isfinite(dT).all(1) & np.isfinite(dP).all(1)
    dT, dP = dT[ok], dP[ok]
    print(f"{ok.sum()} usable daily moves  ({TARGET} vs {PEER})")

    atm = int(np.argmin((gpts[:, 0] - ATM_K) ** 2 + (gpts[:, 1] - ATM_T) ** 2))
    c_lead = _corr_to_vector(dP[:, atm], dT).reshape(KK.shape)               # (1) peer ATM -> target
    c_point = np.array([np.corrcoef(dT[:, g], dP[:, g])[0, 1]                # (2) point-by-point
                        for g in range(len(gpts))]).reshape(KK.shape)

    ik, it = int(np.argmin(np.abs(KS))), int(np.argmin(np.abs(TS - ATM_T)))
    for lab, C in [(f"{PEER}@ATM -> {TARGET} surface", c_lead),
                   (f"{TARGET}<->{PEER} point-by-point", c_point)]:
        wing = float(np.nanmean([C[0], C[-1]]))                              # |k|=0.4 columns
        print(f"  {lab:34s} ATM={C[ik, it]:.2f}   wing(|k|=0.4)={wing:.2f}")

    titles = [f"corr( {PEER} ΔIV @ k={ATM_K},T={ATM_T} ,  {TARGET} ΔIV everywhere )",
              f"corr( {TARGET} ΔIV ,  {PEER} ΔIV )   point-by-point"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, C in zip(axes, titles, [c_lead, c_point]):
        im = ax.imshow(C.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=[KS[0], KS[-1], 0, len(TS) - 1])
        ax.set_yticks(range(len(TS))); ax.set_yticklabels([f"{t:.2f}" for t in TS])
        ax.set_xlabel(f"{TARGET} log-moneyness k"); ax.set_ylabel("maturity T")
        ax.set_title(title, fontsize=9); ax.axvline(0, color="w", ls="--", lw=0.8)
        fig.colorbar(im, ax=ax, label="correlation of daily IV moves")
    fig.suptitle(f"Cross-asset comovement: level factor real, wing coupling spurious "
                 f"({TARGET} vs {PEER}, {ok.sum()} days)", fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
