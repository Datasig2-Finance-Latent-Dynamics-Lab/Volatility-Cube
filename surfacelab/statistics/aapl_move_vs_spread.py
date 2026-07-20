"""
AAPL daily move vs bid-ask spread, in vol points -- at the money AND in the wings.

The point: a cross-asset link can only add the part of AAPL's daily move it can
anticipate from the market (SPY). At the money the move is large and well outside
the spread, but it is also redundant with AAPL's own quote; in the wings the part
SPY explains is tiny (low correlation) and sits inside a wide spread.

Everything is computed on the same fixed (k,T) grid the cross-asset matrix uses, so
the name-vs-SPY correlation here matches that figure (ATM ~0.73, wings ~0.16).

Run:
    .venv/bin/python3 -m surfacelab.statistics.aapl_move_vs_spread
"""

import numpy as np
from scipy.stats import norm

from surfacelab.experiments.configs import _market_thesis
from surfacelab.statistics.cross_asset_corr import _surface_series, KS, TS


def _grid():
    KK, TT = np.meshgrid(KS, TS, indexing="ij")
    return np.stack([KK.ravel(), TT.ravel()], axis=1), KK.ravel()


def _iv_spread(ds, t, mask):
    """IV bid-ask spread (vol points) for masked quotes on day t, via normalised BS vega
    iv_spread = (ask-bid)/(sqrt(T) phi(d1)) -- same conversion as models/cnp/trainer.py."""
    k = ds.query_feats[t, mask, 0]
    T = ds.query_feats[t, mask, 1]
    iv = ds.targets[t, mask]
    bid, ask = ds.bid[t, mask], ds.ask[t, mask]
    ok = (T > 0) & np.isfinite(bid) & np.isfinite(ask) & np.isfinite(iv) & (iv > 0) & ((ask - bid) > 0)
    if not ok.any():
        return np.array([])
    sqT = np.sqrt(np.maximum(T[ok], 1e-12))
    d1 = (-k[ok] + 0.5 * iv[ok] ** 2 * T[ok]) / (iv[ok] * sqT + 1e-14)
    vega = sqT * norm.pdf(d1)
    return (ask[ok] - bid[ok]) / np.maximum(vega, 1e-8)


def _corr_cols_to_vec(Y, x):
    """Pearson corr of each column of Y:(n,G) with x:(n,) -> (G,)."""
    x = x - x.mean()
    Y = Y - Y.mean(0)
    return (Y.T @ x) / (np.sqrt((Y * Y).sum(0)) * np.sqrt((x * x).sum()) + 1e-12)


def main():
    ds, _ = _market_thesis()()          # last 900 market days, same window as the xcorr matrix
    names = ds.meta["asset_names"]
    aapl, spy = names.index("AAPL"), names.index("SPY")
    gpts, kcol = _grid()

    # grid-interpolated surfaces -> daily increments on complete days
    A = _surface_series(ds, aapl, gpts)
    S = _surface_series(ds, spy, gpts)
    dA, dS = np.diff(A, axis=0), np.diff(S, axis=0)
    ok = np.isfinite(dA).all(1) & np.isfinite(dS).all(1)
    dA, dS = dA[ok], dS[ok]

    atm_pt = int(np.argmin(np.abs(kcol)))            # grid column at k=0
    spy_atm = dS[:, atm_pt]                           # SPY ATM increment = market-factor proxy

    rho_lead = _corr_cols_to_vec(dA, spy_atm)         # (1) AAPL[grid] vs SPY[ATM]: the level factor
    rho_point = np.array([np.corrcoef(dA[:, g], dS[:, g])[0, 1]   # (2) AAPL vs SPY at SAME (k,T)
                          for g in range(dA.shape[1])])
    move_std = dA.std(0)                              # AAPL daily move std, per grid point
    vp = 100.0

    def region(label, sel, spread_lo, spread_hi):
        r_lead = float(np.mean(rho_lead[sel]))        # market-level coupling (redundant w/ own ATM quote)
        r_point = float(np.mean(rho_point[sel]))      # peer-specific coupling (the only thing a link adds)
        mv = float(np.sqrt(np.mean(move_std[sel] ** 2)))     # RMS daily move over region
        lvl_part = abs(r_lead) * mv                   # part tied to the market level
        peer_part = abs(r_point) * mv                 # part a cross-asset link could uniquely anticipate
        # spread over actual quotes in this |k| band
        sp = []
        for t in range(ds.n_days):
            m = ds.asset_ids[t] == aapl
            kk = ds.query_feats[t, :, 0]
            sp.append(_iv_spread(ds, t, m & (np.abs(kk) >= spread_lo) & (np.abs(kk) < spread_hi)))
        sp = np.concatenate(sp) if sp else np.array([np.nan])
        sp_med = float(np.median(sp))
        print(f"\n[{label}]  grid pts={int(sel.sum())}  quotes={sp.size}")
        print(f"  corr  level factor (SPY ATM)     : {r_lead:6.3f}   <- redundant with AAPL's own ATM quote")
        print(f"  corr  point-by-point (SPY same k): {r_point:6.3f}   <- the only thing a peer link adds")
        print(f"  daily move  RMS std              : {mv*vp:6.2f} vp")
        print(f"  level-tied part   (r_lead*move)  : {lvl_part*vp:6.2f} vp")
        print(f"  peer-specific part (r_point*move): {peer_part*vp:6.2f} vp")
        print(f"  bid-ask spread  median           : {sp_med*vp:6.2f} vp   (mean {np.mean(sp)*vp:.2f})")
        print(f"  peer-specific part / spread (med): {peer_part/sp_med:6.2f}")
        print(f"  peer-specific part / spread(mean): {peer_part/np.mean(sp):6.2f}")

    print("=" * 60)
    print(f"AAPL move vs spread, by region   (days={dA.shape[0]}, vol points)")
    print("=" * 60)
    region("AT THE MONEY  |k|<0.05", np.abs(kcol) < 0.05, 0.0, 0.05)
    region("WINGS  |k|>=0.30",       np.abs(kcol) >= 0.30, 0.30, 99.0)
    print("=" * 60)


if __name__ == "__main__":
    main()
