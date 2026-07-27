from __future__ import annotations
import numpy as np
import duckdb
from scipy.stats import norm
from surfacelab.data.dataset import Dataset as SurfaceDataset

# NFLX starts 2025-11-17; NVDA starts 2024-06-10 — including them collapses the
# usable date range to <=490 dates. Default to the 8 assets present since Aug 2022.
ASSETS = ["AAPL", "AMD", "AMZN", "GOOGL", "META", "MSFT", "SPY", "TSLA"]
ASSETS_ALL = ["AAPL", "AMD", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "SPY", "TSLA"]

_VAL_FRAC  = 0.15     # fraction of dates used for validation (chronological split)
_IV_MAX    = 3.0      # cap on implied volatility — AMD has deep-ITM calls up to ~5.0
_ATM_BAND  = 0.15     # |log(K/F_stored)| window for the parity forward estimate
_MIN_PAIRS = 3        # paired call/put strikes required to trust an implied forward


# ── Black-76 implied vol (vectorised) ─────────────────────────────────────────
def _black76_call(F, K, T, sig):
    """Undiscounted (forward-measure) Black-76 call price."""
    sq = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig * sig * T) / sq
    d2 = d1 - sq
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def _implied_vol(price, is_call, F, K, T, iv_max, n_iter=64):
    """Invert Black-76 for an array of options by bisection.

    `price` is the *undiscounted* (forward) option price.  Puts are mapped to the
    equivalent call via put-call parity (call = put + (F - K)) so a single solver
    handles both.  Returns NaN where the price violates the no-arbitrage bounds.
    """
    F = np.asarray(F, float); K = np.asarray(K, float)
    T = np.asarray(T, float); price = np.asarray(price, float)
    is_call = np.asarray(is_call, bool)

    call_price = np.where(is_call, price, price + (F - K))
    intrinsic  = np.maximum(F - K, 0.0)
    valid = (T > 0) & (F > 0) & (K > 0) & \
            (call_price > intrinsic + 1e-8) & (call_price < F - 1e-8)

    hi0 = max(iv_max * 1.5, 5.0)
    lo  = np.full(F.shape, 1e-4)
    hi  = np.full(F.shape, hi0)
    Fs = np.where(F > 0, F, 1.0); Ks = np.where(K > 0, K, 1.0)
    Ts = np.where(T > 0, T, 1.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        pr  = _black76_call(Fs, Ks, Ts, mid)
        hi  = np.where(pr > call_price, mid, hi)
        lo  = np.where(pr <= call_price, mid, lo)
    iv = 0.5 * (lo + hi)
    return np.where(valid, iv, np.nan)


def load_grouptech(
    csv_path: str,
    n_train_days: int | None = None,
    n_val_days: int | None = None,
    seed: int = 0,
    val_frac: float = _VAL_FRAC,
    assets: list[str] | None = None,
    ctx_frac: float = 0.6,
    iv_max: float = _IV_MAX,
    n_eval_days: int | None = None,
    obs_col: str = "date",
    quote_side: str = "otm",
) -> SurfaceDataset:
    """Load Group Tech US options data from CSV into a Dataset.

    Implied vols are recomputed from a *parity-implied forward* before use.  The
    raw feed's stored ``forward`` is biased for the single names (dividends / borrow
    are mishandled — off by ~0.1–0.2%, vs ~0.02% for SPY), which pushes the OTM-put
    and OTM-call wings apart and produces a visible jump at the money once the two
    sides are spliced.  For each (observation, asset, maturity) we re-estimate the
    forward from put-call parity over the near-ATM strikes that quote *both* a call
    and a put — ``F = (C - P)/discount + K`` — take the robust (median) estimate,
    then re-invert Black-76 against that forward so the two wings line up.  Strikes
    are re-expressed as ``log(K / F_implied)``.

    quote_side : which option type supplies the IV at each strike (after the forward
        is implied and IV re-inverted consistently for *both* types):
        "otm"  — out-of-the-money quotes (the liquid side): puts for negative
                 log-moneyness (k = log K/F < 0), calls for non-negative.  Default.
                 Deep-ITM American options carry an early-exercise premium that the
                 European Black-76 inversion misreads as extra vol, so OTM is the
                 clean side — and the standard remedy for spiky illiquid smiles.
        "call" — calls only;  "put" — puts only.  (Diagnostic; same implied-forward
                 IVs, just restricted to one type — note these still mix ITM quotes.)
      A call and a put at the same strike share the same log-moneyness, so the "otm"
      partition selects exactly one row per (strike, maturity) — no duplicates.

    Only quotes with a positive bid and ask are used.  Observations where any of the
    configured assets has no data are dropped so every observation has a complete,
    consistent set of assets.  All quality filters are pushed into DuckDB so the
    full 1.7 GB CSV is never materialised.

    obs_col : "date" (daily, one surface per trading day) or "datetime" (hourly).
    n_eval_days : if set, the last n_eval_days observations become the val split.
    """
    if quote_side not in ("otm", "call", "put"):
        raise ValueError(
            f"quote_side must be 'otm', 'call' or 'put'; got {quote_side!r}")

    if assets is None:
        assets = ASSETS
    asset_to_id = {a: i for i, a in enumerate(assets)}

    # Pull BOTH calls and puts: the implied forward needs paired strikes.  IV and
    # log-moneyness are recomputed below, so we don't filter on the stored values.
    base_cols = ["underlying", "type", "strike", "T", "bid", "ask",
                 "forward", "discount"]
    usecols = list(dict.fromkeys([obs_col] + base_cols))  # obs_col first, no dupes

    assets_sql = ", ".join(f"'{a}'" for a in assets)
    cols_sql   = ", ".join(f'"{c}"' for c in usecols)
    df = duckdb.query(f"""
        SELECT {cols_sql}
        FROM read_csv_auto('{csv_path}')
        WHERE type IN ('call', 'put')
          AND underlying IN ({assets_sql})
          AND bid > 0
          AND ask > 0
          AND strike IS NOT NULL
          AND T IS NOT NULL
          AND forward IS NOT NULL
          AND discount IS NOT NULL
    """).df()

    df["mid"] = 0.5 * (df["bid"] + df["ask"])

    # ── parity-implied forward per (obs, asset, maturity) ─────────────────────
    calls = df[df["type"] == "call"][[obs_col, "underlying", "T", "strike",
                                      "mid", "discount", "forward"]]
    puts  = df[df["type"] == "put"][[obs_col, "underlying", "T", "strike", "mid"]]
    pair = calls.merge(puts, on=[obs_col, "underlying", "T", "strike"],
                       suffixes=("_c", "_p"))
    # Restrict to near-ATM strikes where parity is most reliable (tight spreads,
    # negligible early-exercise premium).
    k_stored = np.log(pair["strike"].values / pair["forward"].values)
    pair = pair[np.abs(k_stored) < _ATM_BAND]
    # Each paired strike implies F = (C - P)/d + K; the median is robust to noise.
    pair = pair.assign(
        F_est=(pair["mid_c"] - pair["mid_p"]) / pair["discount"] + pair["strike"])
    g = pair.groupby([obs_col, "underlying", "T"])["F_est"].agg(["median", "count"])
    g = g.rename(columns={"median": "F_impl"}).reset_index()
    g.loc[g["count"] < _MIN_PAIRS, "F_impl"] = np.nan  # too few pairs → fall back

    df = df.merge(g[[obs_col, "underlying", "T", "F_impl"]],
                  on=[obs_col, "underlying", "T"], how="left")
    # Fall back to the stored forward where parity couldn't be implied.
    F = df["F_impl"].to_numpy()
    F = np.where(np.isfinite(F) & (F > 0), F, df["forward"].to_numpy())
    df["F"] = F

    # ── re-strike and re-invert IV against the implied forward ────────────────
    df["lm"] = np.log(df["strike"].to_numpy() / F)
    is_call = (df["type"] == "call").to_numpy()
    # OTM partition (or single-side diagnostic), on the corrected moneyness.
    if quote_side == "otm":
        keep = (is_call & (df["lm"].to_numpy() >= 0.0)) | \
               (~is_call & (df["lm"].to_numpy() < 0.0))
    elif quote_side == "call":
        keep = is_call
    else:
        keep = ~is_call
    df = df[keep].reset_index(drop=True)

    fwd_price = (df["mid"].to_numpy() / df["discount"].to_numpy())  # forward-measure
    iv = _implied_vol(fwd_price, (df["type"] == "call").to_numpy(),
                      df["F"].to_numpy(), df["strike"].to_numpy(),
                      df["T"].to_numpy(), iv_max)
    df["iv"] = iv
    df = df[np.isfinite(iv) & (iv > 0.0) & (iv <= iv_max)].reset_index(drop=True)

    # logmoneyness column now carries the parity-corrected moneyness.
    df["logmoneyness"] = df["lm"]

    # Drop observations that don't have data for every configured asset
    assets_set = set(assets)
    obs_coverage = df.groupby(obs_col)["underlying"].apply(
        lambda s: assets_set.issubset(set(s))
    )
    complete_obs = set(obs_coverage[obs_coverage].index)
    df = df[df[obs_col].isin(complete_obs)]

    # Chronological split — lexicographic sort works for "YYYY-MM-DD" and
    # "YYYY-MM-DD HH:00" alike.
    dates = sorted(df[obs_col].unique())
    n_dates = len(dates)
    if n_eval_days is not None:
        if n_eval_days >= n_dates:
            raise ValueError(
                f"n_eval_days={n_eval_days} must be less than total obs={n_dates}")
        val_dates_set   = set(dates[-n_eval_days:])
        train_dates_set = set(dates[:-n_eval_days])
    else:
        n_val = max(1, int(n_dates * val_frac))
        val_dates_set   = set(dates[-n_val:])
        train_dates_set = set(dates[:-n_val])

    rng = np.random.default_rng(seed)
    train_dates_sorted = sorted(train_dates_set)
    val_dates_sorted   = sorted(val_dates_set)

    if n_train_days is not None and n_train_days < len(train_dates_sorted):
        train_dates_sorted = sorted(
            rng.choice(train_dates_sorted, n_train_days, replace=False).tolist())
    if n_val_days is not None and n_val_days < len(val_dates_sorted):
        val_dates_sorted = sorted(
            rng.choice(val_dates_sorted, n_val_days, replace=False).tolist())

    kept_dates  = train_dates_sorted + val_dates_sorted
    split_label = [0] * len(train_dates_sorted) + [1] * len(val_dates_sorted)

    # Per-day per-asset log forward (constant across strikes for a given asset/date)
    fwd_pivot = (
        df.groupby([obs_col, "underlying"])["F"]
        .first()
        .unstack(level="underlying")
        .reindex(index=kept_dates, columns=assets)
    )
    log_fwd_arr = np.log(fwd_pivot.values.astype(np.float32))  # (n_days, n_assets)

    # Sort by (T, logmoneyness) so all assets interleave by maturity/moneyness,
    # guaranteeing every asset appears in both the context pool (first ctx_frac)
    # and the target pool (remainder).
    day_lm, day_T, day_aid, day_iv = [], [], [], []
    day_bid_norm, day_ask_norm, day_is_call = [], [], []
    for d in kept_dates:
        sub = df[df[obs_col] == d].sort_values(["T", "logmoneyness"])
        day_lm.append(sub["logmoneyness"].values.astype(np.float32))
        day_T.append(sub["T"].values.astype(np.float32))
        day_aid.append(sub["underlying"].map(asset_to_id).values.astype(np.int64))
        day_iv.append(sub["iv"].values.astype(np.float32))
        fwd_disc = (sub["F"] * sub["discount"]).values
        fwd_disc = np.where(fwd_disc > 0, fwd_disc, np.nan)
        day_bid_norm.append((sub["bid"].values / fwd_disc).astype(np.float32))
        day_ask_norm.append((sub["ask"].values / fwd_disc).astype(np.float32))
        day_is_call.append((sub["type"].values == "call"))

    del df

    max_pts = max(len(x) for x in day_lm)
    n_days  = len(kept_dates)

    lm_arr       = np.zeros((n_days, max_pts), dtype=np.float32)
    T_arr        = np.zeros((n_days, max_pts), dtype=np.float32)
    aid_arr      = np.zeros((n_days, max_pts), dtype=np.int64)
    iv_arr       = np.zeros((n_days, max_pts), dtype=np.float32)
    bid_norm_arr = np.full((n_days, max_pts), np.nan, dtype=np.float32)
    ask_norm_arr = np.full((n_days, max_pts), np.nan, dtype=np.float32)
    is_call_arr  = np.zeros((n_days, max_pts), dtype=bool)

    for i, (lm, t, aid, iv, bid_n, ask_n, isc) in enumerate(
        zip(day_lm, day_T, day_aid, day_iv, day_bid_norm, day_ask_norm, day_is_call)
    ):
        n = len(lm)
        lm_arr[i, :n]       = lm
        T_arr[i, :n]        = t
        aid_arr[i, :n]      = aid
        iv_arr[i, :n]       = iv
        bid_norm_arr[i, :n] = bid_n
        ask_norm_arr[i, :n] = ask_n
        is_call_arr[i, :n]  = isc

    ctx_max = max(1, int(max_pts * ctx_frac))

    return SurfaceDataset(
        query_feats=np.stack([lm_arr, T_arr], axis=-1),
        asset_ids=aid_arr,
        targets=iv_arr,
        split=np.array(split_label, dtype=np.int8),
        ctx_max=ctx_max,
        n_assets=len(assets),
        params=None,
        meta={
            "query_feat_names": ["lm", "T"],
            "target_name": "IV",
            "dgp": "grouptech_market",
            "asset_names": assets,
            "dates": kept_dates,
            "obs_col": obs_col,
            "log_fwd": log_fwd_arr,
        },
        bid=bid_norm_arr,
        ask=ask_norm_arr,
        is_call=is_call_arr,
    )
