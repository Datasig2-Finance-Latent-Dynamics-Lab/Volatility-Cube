"""
Sequential decay with RANDOM intermittent gaps in the target (Alvaro's variant).

Like seq_asymmetric.py, but the target (AAPL) gets a RANDOM number of quotes each day —
0 w.p. 0.5, 1 w.p. 0.3, 2 w.p. 0.2 — while peers are always fully observed. On the many
zero-quote days there is no re-anchor, so the filter must coast on the peers (cross-asset)
or on a stale own state (AAPL-only). This is the regime where cross-asset coupling should
finally pay off; we break the error down by whether AAPL was seen that day and by how many
days since it was last seen (gap length).

Run:  .venv/bin/python3 -m surfacelab.statistics.seq_gaps
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from surfacelab.models.kalman_ssvi import KalmanSSVIModel
from surfacelab.statistics.seq_asymmetric import build_split, OUT, TARGET

DIST = {0: 0.50, 1: 0.30, 2: 0.20}     # P(#AAPL quotes today)


def run_seq_random(model, ds, seed=0):
    """Free-run; AAPL quote count drawn from DIST each day, peers full. Carry posterior fwd.
    Returns rows (day_index, AAPL_rmse_volpts, k_quotes, gap_since_last_seen)."""
    rng = np.random.default_rng(seed)
    ks, ps = np.array(list(DIST)), np.array(list(DIST.values()))
    names = ds.meta["asset_names"]; ex = names.index(TARGET)
    test = np.sort(np.where(ds.split == 1)[0])
    model.reset_sequence()
    model.seed_prior(ds.quotes_at(test[0] - 1))            # perfect prior (full prev day)
    out = []; gap = 0
    for t in test:
        t = int(t)
        valid = ds.valid_points(t)
        aid = ds.asset_ids[t, valid]
        peer = valid[aid != ex]
        tgt = valid[aid == ex]
        k = int(rng.choice(ks, p=ps))
        k = min(k, len(tgt))
        ti = rng.choice(tgt, k, replace=False) if k else np.empty(0, int)
        ci = np.concatenate([peer, ti])
        q_all = ds.query_at(t, valid)
        pred = model.step(ds.quotes_at(t, ci), q_all).iv
        gap = 0 if k > 0 else gap + 1
        m = aid == ex
        if m.any():
            true = ds.targets[t, valid][m]
            out.append((t, float(np.sqrt(np.mean((pred[m] - true) ** 2)) * 100), k, gap))
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = build_split()
    print(f"AAPL quotes/day ~ {DIST} (E={sum(k*p for k,p in DIST.items()):.2f}/day); peers full")
    res = {}
    for mode in ("increments", "levels"):
        for cross in (True, False):
            m = KalmanSSVIModel(transition_mode=mode, cross_asset=cross)
            print(f"training {m.name} …", flush=True)
            m.train(ds, saved=True)
            res[(mode, cross)] = run_seq_random(m, ds)

    print("\n=== steady-state AAPL RMSE (vol pts), 2nd half ===")
    print(f"{'mode':<12}{'all days':>20}{'0-quote days':>20}{'>=1-quote days':>20}")
    print(f"{'':12}{'cross  nox  gain':>20}{'cross  nox  gain':>20}{'cross  nox  gain':>20}")
    for mode in ("increments", "levels"):
        c, x = res[(mode, True)], res[(mode, False)]
        h = len(c) // 2
        cell = []
        for msk_name in ("all", "zero", "pos"):
            def sel(r):
                rr = r[h:]
                if msk_name == "zero": rr = rr[rr[:, 2] == 0]
                elif msk_name == "pos": rr = rr[rr[:, 2] > 0]
                return rr[:, 1].mean() if len(rr) else float("nan")
            cc, xx = sel(c), sel(x)
            cell.append(f"{cc:5.2f} {xx:5.2f} {xx-cc:+5.2f}")
        print(f"{mode:<12}{cell[0]:>20}{cell[1]:>20}{cell[2]:>20}")

    # RMSE vs gap length (increments, the well-specified filter)
    print("\n=== increments mode: AAPL RMSE by gap length (days since last quote) ===")
    c, x = res[("increments", True)], res[("increments", False)]
    h = len(c) // 2; cc, xx = c[h:], x[h:]
    print(f"{'gap (days)':<12}{'cross':>8}{'nox':>8}{'gain':>8}{'n_days':>8}")
    for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)]:
        mc = cc[(cc[:, 3] >= lo) & (cc[:, 3] <= hi)]
        mx = xx[(xx[:, 3] >= lo) & (xx[:, 3] <= hi)]
        if len(mc):
            lab = f"{lo}" if lo == hi else f"{lo}-{hi if hi<99 else '+'}"
            print(f"{lab:<12}{mc[:,1].mean():>8.2f}{mx[:,1].mean():>8.2f}"
                  f"{mx[:,1].mean()-mc[:,1].mean():>+8.2f}{len(mc):>8d}")

    _plot(res)
    print(f"\nWrote {OUT}/seq_gaps.png")


def _plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    c, x = res[("increments", True)], res[("increments", False)]
    ax.plot(range(len(c)), c[:, 1], "C0", lw=1.2, label="cross-asset (all)")
    ax.plot(range(len(x)), x[:, 1], "C3", lw=1.2, label="AAPL-only")
    zero = c[c[:, 2] == 0]
    ax.scatter([np.where(c[:, 0] == z)[0][0] for z in zero[:, 0]], zero[:, 1],
               s=8, c="C0", alpha=0.4, label="0-quote day (cross)")
    ax.set(title="Kalman SSVI (increments) — AAPL with random gaps, peers full",
           xlabel="days into free-run", ylabel="AAPL RMSE (vol points)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "seq_gaps.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
