"""
Black-Scholes pricing utilities shared across dgraph and neural_processes experiments.

Convention: logmoneyness = log(K / F), so K = F * exp(lm).
Setting forward=1 and discount=1 returns the forward-normalized call price C / (F * discount),
which is purely a function of (iv, lm, T) and is used throughout when actual
forward/discount data is not available.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_call_from_iv(
    iv: float | np.ndarray,
    logmoneyness: float | np.ndarray,
    T: float | np.ndarray,
    forward: float | np.ndarray = 1.0,
    discount: float | np.ndarray = 1.0,
) -> np.ndarray:
    """
    Black-Scholes call price from implied vol and log-moneyness.

    Parameters
    ----------
    iv          : implied volatility (annualised, decimal, e.g. 0.20)
    logmoneyness: log(K/F)
    T           : time to expiry in years
    forward     : undiscounted forward price F = S * exp(r*T)
    discount    : discount factor exp(-r*T)

    Returns
    -------
    call price  = discount * forward * (N(d1) - exp(lm) * N(d2))
    """
    iv  = np.asarray(iv,  dtype=float)
    lm  = np.asarray(logmoneyness, dtype=float)
    T   = np.asarray(T,   dtype=float)
    sqT = np.sqrt(np.maximum(T, 1e-12))
    d1  = (-lm + 0.5 * iv ** 2 * T) / (iv * sqT + 1e-14)
    d2  = d1 - iv * sqT
    return discount * forward * (norm.cdf(d1) - np.exp(lm) * norm.cdf(d2))


def bs_iv_from_call(
    call_price: float | np.ndarray,
    logmoneyness: float | np.ndarray,
    T: float | np.ndarray,
    forward: float | np.ndarray = 1.0,
    discount: float | np.ndarray = 1.0,
    n_iter: int = 80,
) -> np.ndarray:
    """
    Newton-Raphson Black-Scholes implied-vol solver.  Inverse of bs_call_from_iv.

    Returns NaN for inputs that are below intrinsic value or otherwise
    uninvertible (e.g. expired options).
    """
    call = np.asarray(call_price,  dtype=float)
    lm   = np.asarray(logmoneyness, dtype=float)
    T    = np.asarray(T,    dtype=float)
    fwd  = np.asarray(forward,  dtype=float)
    disc = np.asarray(discount, dtype=float)
    sqT  = np.sqrt(np.maximum(T, 1e-12))

    intrinsic = disc * fwd * np.maximum(1.0 - np.exp(lm), 0.0)
    valid = call > intrinsic + 1e-10
    iv = np.full_like(call, 0.30)

    for _ in range(n_iter):
        d1   = (-lm + 0.5 * iv ** 2 * T) / (iv * sqT + 1e-14)
        bs   = disc * fwd * (norm.cdf(d1) - np.exp(lm) * norm.cdf(d1 - iv * sqT))
        vega = disc * fwd * norm.pdf(d1) * sqT
        mask = valid & (vega > 1e-14)
        iv   = np.where(mask, iv - (bs - call) / vega, iv)
        iv   = np.where(valid, np.clip(iv, 1e-8, 10.0), iv)

    final = bs_call_from_iv(iv, lm, T, fwd, disc)
    bad   = ~valid | (np.abs(final - call) > 1e-5)
    return np.where(bad, np.nan, iv)
