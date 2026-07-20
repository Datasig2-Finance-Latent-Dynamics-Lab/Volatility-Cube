"""
kalman_pca in the illiquid sequential regime, with the weight in the FILTER UPDATE (not the
basis). The PCA basis is UNWEIGHTED — it learns the true joint multi-asset modes — and AAPL's
few daily quotes are up-weighted only when estimating the factor scores, so they aren't
swamped by the hundreds of peer quotes. Sweeps AAPL's observation weight and the number of
AAPL quotes/day (1,2,4,8). 100 test days (steady state = 2nd half).

Run:  .venv/bin/python3 -m surfacelab.statistics.seq_kalman_pca
"""
from __future__ import annotations

import numpy as np

from surfacelab.models.kalman import KalmanFactorModel
from surfacelab.models.kalman_ssvi import KalmanSSVIModel
from surfacelab.statistics.seq_asymmetric import build_split, TARGET

MAX_DAYS = 400
SEEDS = [0, 1, 2]


def run_seq(model, ds, n_quotes, use_peers=True, max_days=MAX_DAYS, seed=0):
    rng = np.random.default_rng(seed)
    names = ds.meta["asset_names"]; ex = names.index(TARGET)
    test = np.sort(np.where(ds.split == 1)[0])[:max_days]
    model.reset_sequence()
    model.seed_prior(ds.quotes_at(test[0] - 1))
    err = []
    for t in test:
        t = int(t); valid = ds.valid_points(t); aid = ds.asset_ids[t, valid]
        peer = valid[aid != ex]; tgt = valid[aid == ex]
        k = min(n_quotes, len(tgt))
        ti = rng.choice(tgt, k, replace=False) if k else np.empty(0, int)
        ci = np.concatenate([peer, ti]) if use_peers else ti
        pred = model.step(ds.quotes_at(t, ci), ds.query_at(t, valid)).iv
        m = aid == ex
        if m.any():
            err.append(float(np.sqrt(np.mean((pred[m] - ds.targets[t, valid][m]) ** 2)) * 100))
    return np.mean(err[len(err) // 2:])      # steady-state (2nd half)


def agg(model, ds, n, **kw):
    """Steady-state RMSE averaged over quote-sampling SEEDS → 'mean±std'."""
    v = [run_seq(model, ds, n, seed=s, **kw) for s in SEEDS]
    return np.mean(v), np.std(v)


def fmt(model, ds, Ns, **kw):
    return "".join(f"{m:>6.2f}±{s:.2f}" for m, s in (agg(model, ds, n, **kw) for n in Ns))


def main():
    ds = build_split()
    ex = ds.meta["asset_names"].index(TARGET)
    Ns = [1, 2, 4, 8]
    print(f"{ds.n_days} days, {ds.n_assets} assets; AAPL illiquid, peers full; "
          f"{MAX_DAYS} test days, seeds={SEEDS}\n")

    ssvi = KalmanSSVIModel(transition_mode="increments", cross_asset=True)
    print("training kalman_ssvi_inc …", flush=True); ssvi.train(ds, saved=True)
    pca = KalmanFactorModel(n_components=20)          # UNWEIGHTED basis (true joint modes)
    print("training kalman_pca (unweighted basis) …", flush=True); pca.train(ds, saved=True)
    pca.obs_weight_asset = ex

    print(f"\nAAPL steady-state RMSE (vol pts, mean±std over seeds).  cols = AAPL quotes/day")
    print(f"{'config':<28}" + "".join(f"{n:>11}" for n in Ns))
    pca.obs_weight = 1.0
    print(f"{'kalman_ssvi (peers full)':<28}{fmt(ssvi, ds, Ns)}")
    print(f"{'kalman_pca AAPL-only ctx':<28}{fmt(pca, ds, Ns, use_peers=False)}")
    print("  -- kalman_pca, peers full, AAPL obs up-weighted in the filter update: --")
    for ow in (1, 20, 100):
        pca.obs_weight = float(ow)
        print(f"{'  obs_weight=' + str(ow):<28}{fmt(pca, ds, Ns)}")


if __name__ == "__main__":
    main()
