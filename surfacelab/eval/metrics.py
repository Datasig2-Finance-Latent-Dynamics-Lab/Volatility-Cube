"""
Error metrics and the liquid/illiquid region split.

The liquid region is the near-ATM, short-dated band that is quoted directly; the illiquid
region (wide strikes, long maturities) is what has to be extrapolated systematically.
Splitting RMSE this way is the headline diagnostic: how well does each method extrapolate
into the region it was *not* given?
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def liquid_mask(k, T, k_liq: float = 0.2, T_liq: float = 0.5) -> np.ndarray:
    """True where a point is in the liquid (near-ATM, short-dated) region."""
    return (np.abs(np.asarray(k)) <= k_liq) & (np.asarray(T) <= T_liq)


def error_stats(pred, true) -> dict:
    """Squared- and absolute-error sums + count (so summaries can pool correctly)."""
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    err = pred - true
    return {"sq": float(np.sum(err ** 2)), "abs": float(np.sum(np.abs(err))),
            "n": int(err.size)}


def split_stats(pred, true, k, T, *, k_liq=0.2, T_liq=0.5) -> dict:
    """Full + liquid + illiquid error sums for one prediction batch."""
    liq = liquid_mask(k, T, k_liq, T_liq)
    out = {}
    for tag, m in (("", np.ones_like(liq)), ("_liquid", liq), ("_illiquid", ~liq)):
        if m.any():
            s = error_stats(np.asarray(pred)[m], np.asarray(true)[m])
            out[f"sq{tag}"] = s["sq"]; out[f"n{tag}"] = s["n"]
            if tag == "":
                out["abs"] = s["abs"]
        else:
            out[f"sq{tag}"] = 0.0; out[f"n{tag}"] = 0
    return out


def rmse(sq_sum, n) -> float:
    return float(np.sqrt(sq_sum / n)) if n else float("nan")


# ── call-price diagnostics ──────────────────────────────────────────────────────
def bs_call_norm(iv, k, T) -> np.ndarray:
    """Forward-and-discount-normalised Black-Scholes call price from IV.

    Convention: ``k = log(K / F)`` (matching the dataset).  Returns
    ``C / (F * discount) = N(d1) - exp(k) * N(d2)``, exactly the normalisation the
    dataset stores for ``bid`` / ``ask`` (quote / (forward * discount)), so the two
    are directly comparable.
    """
    iv = np.asarray(iv, float); k = np.asarray(k, float); T = np.asarray(T, float)
    sqT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (-k + 0.5 * iv ** 2 * T) / (iv * sqT + 1e-14)
    d2 = d1 - iv * sqT
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def call_price_stats(pred_iv, k, T, bid, ask, is_call=None) -> dict:
    """Black-Scholes option-price diagnostics vs the forward-normalised bid/ask spread.

    Converts the *predicted* IV to a normalised option price and compares it to the
    observed bid/ask.  The price must match the option type the quote came from: for an
    OTM surface, points with k < 0 are PUTS and k >= 0 are CALLS, so comparing a call
    price against a put's spread (they differ by the parity term 1 - e^k) would flag
    every put as out-of-spread.  `is_call` (per point) selects the right price; the
    normalised put follows from put-call parity p = c - (1 - e^k).  When `is_call` is
    None every point is treated as a call (back-compat / call-only datasets).

    Returns sums (not means) so summaries pool exactly across days the way RMSE does:

        call_sq            : sum of (price_pred - mid)^2   (mid = (bid + ask) / 2)
        call_n             : count of points with a finite (bid, ask)
        call_oob            : count of those points whose price_pred falls outside [bid, ask]
        call_oob_spread_sum : sum of the distance past the nearest spread bound measured in
                              SPREAD WIDTHS, dist / (ask - bid) (0 inside the spread).
                              Dimensionless and comparable across assets/strikes regardless
                              of price level; pools into a MEAN spread-width over the
                              out-of-spread points (divide by call_oob — see records.summary).
                              Mean, not RMS: the ratio has a fat tail (tiny spreads), which an
                              RMS would be dominated by.

    Points with a missing bid/ask are dropped.  Returns zero-counts if none are valid
    (e.g. a synthetic DGP that carries no quotes), so the column is simply absent there.
    """
    if bid is None or ask is None:
        return {"call_sq": 0.0, "call_n": 0, "call_oob": 0, "call_oob_spread_sum": 0.0}
    bid = np.asarray(bid, float); ask = np.asarray(ask, float)
    m = np.isfinite(bid) & np.isfinite(ask)
    if not m.any():
        return {"call_sq": 0.0, "call_n": 0, "call_oob": 0, "call_oob_spread_sum": 0.0}
    k_m = np.asarray(k, float)[m]
    c = bs_call_norm(np.asarray(pred_iv, float)[m], k_m, np.asarray(T, float)[m])
    if is_call is None:
        price = c
    else:
        is_call_m = np.asarray(is_call, bool)[m]
        # normalised put via put-call parity: P/(F*disc) = C/(F*disc) - (1 - e^k)
        price = np.where(is_call_m, c, c - (1.0 - np.exp(k_m)))
    bid, ask = bid[m], ask[m]
    mid = 0.5 * (bid + ask)
    spread = ask - bid
    oob = (price < bid) | (price > ask)
    # signed distance past the nearest bound, floored at 0 (bid <= ask, so at most
    # one of (bid - price), (price - ask) is positive): inside the spread -> 0 error.
    oob_dist = np.maximum(0.0, np.maximum(bid - price, price - ask))
    # the same distance expressed in spread-widths (dimensionless); guard a zero spread.
    oob_spread = oob_dist / np.maximum(spread, 1e-12)
    return {"call_sq": float(np.sum((price - mid) ** 2)),
            "call_n": int(m.sum()),
            "call_oob": int(np.sum(oob)),
            "call_oob_spread_sum": float(np.sum(oob_spread))}
