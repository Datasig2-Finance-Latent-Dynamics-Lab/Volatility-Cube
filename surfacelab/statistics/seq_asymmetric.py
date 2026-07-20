"""
Sequential asymmetric-liquidity decay experiment (Alvaro's design).

Train a filter on the first 500 days (all assets, full info). Then free-run days 500..899:
every day the TARGET (AAPL) gets only N sparse quotes while every peer is fully observed;
the model carries its own posterior forward as tomorrow's prior. The day-500 seed is a
perfect prior, so it decays over ~400 steps and we read the STEADY-STATE target error.

Compares, with everything else identical (same training, same dynamics, same fitting):
  * kalman_ssvi      cross_asset=True  — uses peers' state to hold the target on track
  * kalman_ssvi_nox  cross_asset=False — block-diagonal A AND Q: a pure single-asset filter

If cross-asset coupling is worth anything for an illiquid name, the coupled filter should
hold a lower steady-state AAPL RMSE than the AAPL-only one. SSVI (6 params, smooth, all
maturities) is used precisely because B-splines generalise terribly from a few quotes.

Run:  .venv/bin/python3 -m surfacelab.statistics.seq_asymmetric
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from surfacelab.data import load_grouptech
from surfacelab.experiments.configs import MARKET_CSV
from surfacelab.models.kalman_ssvi import KalmanSSVIModel

OUT = Path(__file__).resolve().parents[2] / "results" / "diagnostics" / "exploitability"
N_TRAIN = 500
N_TEST = 400
TARGET = "AAPL"


def build_split(n_train=N_TRAIN, n_test=N_TEST):
    """First n_train days = train (split 0), next n_test = sequential test (split 1)."""
    ds = load_grouptech(MARKET_CSV)
    keep = min(ds.n_days, n_train + n_test)
    ds = ds.subset(np.arange(keep))
    ds.split = np.where(np.arange(ds.n_days) < n_train, 0, 1).astype(np.int8)
    return ds


def run_seq(model, ds, n_quotes, seed=0):
    """Free-run the filter over the test window; return per-day AAPL RMSE (vol points).

    Peers fully observed each day, AAPL given `n_quotes` of its own; model.step carries the
    posterior forward. Seeded once with the last full (train) day = the perfect prior."""
    rng = np.random.default_rng(seed)
    names = ds.meta["asset_names"]; ex = names.index(TARGET)
    test = np.sort(np.where(ds.split == 1)[0])
    model.reset_sequence()
    model.seed_prior(ds.quotes_at(test[0] - 1))            # perfect prior (full prev day)
    out = []
    for t in test:
        t = int(t)
        valid = ds.valid_points(t)
        aid = ds.asset_ids[t, valid]
        peer = valid[aid != ex]                            # ALL peer quotes (liquid)
        tgt = valid[aid == ex]
        ti = rng.choice(tgt, min(n_quotes, len(tgt)), replace=False) if len(tgt) else np.empty(0, int)
        ci = np.concatenate([peer, ti])
        q_all = ds.query_at(t, valid)
        pred = model.step(ds.quotes_at(t, ci), q_all).iv
        m = aid == ex
        if m.any():
            true = ds.targets[t, valid][m]
            out.append((t, float(np.sqrt(np.mean((pred[m] - true) ** 2)) * 100)))  # vol pts
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = build_split()
    print(f"{ds.n_days} days ({N_TRAIN} train / rest test), {ds.n_assets} assets: {ds.meta['asset_names']}")

    results = {}
    for mode in ("levels", "increments"):
        for cross in (True, False):
            m = KalmanSSVIModel(transition_mode=mode, cross_asset=cross)
            print(f"training {m.name} ({mode}) …", flush=True)
            m.train(ds, saved=True)
            for N in (1, 3):
                key = (mode, cross, N)
                r = run_seq(m, ds, N)
                tail = r[len(r) // 2:, 1]        # steady state = 2nd half (perfect prior decayed)
                results[key] = r
                print(f"  {m.name:18s} N={N}: steady-state AAPL RMSE = {tail.mean():.2f} vol pts "
                      f"(first day {r[0,1]:.2f}, last {r[-1,1]:.2f})", flush=True)

    # summary table: cross vs nox steady-state, and the cross-asset gain
    print("\n=== STEADY-STATE AAPL RMSE (vol points), 2nd-half mean ===")
    print(f"{'mode':<12}{'N':>3}{'cross':>9}{'nox(AAPL-only)':>16}{'gain':>8}")
    for mode in ("levels", "increments"):
        for N in (1, 3):
            c = results[(mode, True, N)][len(results[(mode, True, N)]) // 2:, 1].mean()
            x = results[(mode, False, N)][len(results[(mode, False, N)]) // 2:, 1].mean()
            print(f"{mode:<12}{N:>3}{c:>9.2f}{x:>16.2f}{x - c:>+8.2f}")

    _plot(results)
    print(f"\nWrote {OUT}/seq_asymmetric.png")


def _plot(results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, mode in zip(axes, ("levels", "increments")):
        for cross, lbl, c in ((True, "cross-asset (all)", "C0"), (False, "AAPL-only", "C3")):
            r = results[(mode, cross, 1)]
            ax.plot(range(len(r)), r[:, 1], color=c, label=lbl, lw=1.3)
        ax.set(title=f"Kalman SSVI ({mode}) — AAPL, 1 quote/day, peers full",
               xlabel="days into free-run", ylabel="AAPL RMSE (vol points)")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "seq_asymmetric.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
