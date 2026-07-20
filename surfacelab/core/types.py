"""
Core data types flowing through the model contract.

A `Quotes` holds the observed implied-vol quotes on ONE day, with all assets mixed
together (each point tagged by ``asset_id``).  A `QueryPoints` holds the coordinates
to predict on that same day.  `predict(context: Quotes, query: QueryPoints)` returns
a `SurfacePrediction`.

Everything is single-day: the eval harness loops over days and carries any temporal
state inside the model (see `surfacelab.core.model.SurfaceModel`).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _as1d(x, dtype=np.float64) -> np.ndarray:
    """Coerce to a flat 1-D array of `dtype` — the shared __post_init__ normalisation."""
    return np.asarray(x, dtype=dtype).ravel()


@dataclass
class Quotes:
    """Observed IV quotes on one day, all assets mixed.

    k        : (N,) log-moneyness
    T        : (N,) maturity in years
    iv       : (N,) implied vol
    asset_id : (N,) int, 0-based
    bid, ask : (N,) optional forward-normalised quotes
    """

    k: np.ndarray
    T: np.ndarray
    iv: np.ndarray
    asset_id: np.ndarray
    bid: np.ndarray | None = None
    ask: np.ndarray | None = None

    def __post_init__(self):
        self.k = _as1d(self.k)
        self.T = _as1d(self.T)
        self.iv = _as1d(self.iv)
        self.asset_id = _as1d(self.asset_id, dtype=np.int64)

    @property
    def n(self) -> int:
        return self.k.shape[0]

    @property
    def feats(self) -> np.ndarray:
        """(N, 2) = stack([k, T]) — the model feature ('Q-dim') matrix."""
        return np.stack([self.k, self.T], axis=-1)


@dataclass
class QueryPoints:
    """Coordinates to predict on one day, all assets mixed.

    prior_iv : (N,) optional — yesterday's surface evaluated at these points, used by
               increment/delta models.  NaN where unavailable.
    """

    k: np.ndarray
    T: np.ndarray
    asset_id: np.ndarray
    prior_iv: np.ndarray | None = None

    def __post_init__(self):
        self.k = _as1d(self.k)
        self.T = _as1d(self.T)
        self.asset_id = _as1d(self.asset_id, dtype=np.int64)
        if self.prior_iv is not None:
            self.prior_iv = _as1d(self.prior_iv)

    @property
    def n(self) -> int:
        return self.k.shape[0]

    @property
    def feats(self) -> np.ndarray:
        return np.stack([self.k, self.T], axis=-1)


@dataclass
class SurfacePrediction:
    """Predicted IV at the query points (same order as the QueryPoints)."""

    iv: np.ndarray
    iv_std: np.ndarray | None = None

    def __post_init__(self):
        self.iv = _as1d(self.iv)
        if self.iv_std is not None:
            self.iv_std = _as1d(self.iv_std)


def query_as_quotes(query: QueryPoints, fill_iv: float = 0.2) -> Quotes:
    """Treat query points as Quotes with a flat placeholder IV — the fallback when a model
    is asked to predict without any seeded prior or context (e.g. a bare smoke test)."""
    return Quotes(query.k, query.T, np.full(query.n, fill_iv), query.asset_id)


def to_batch_arrays(context: Quotes, query: QueryPoints):
    """Convert (Quotes, QueryPoints) → the legacy batched-array predict signature.

    Returns (obs_feats, obs_aids, obs_tgts, qry_feats, qry_aids) each with a leading
    batch dim of 1, matching `FittedCNP.predict` and the `SurfacePredictor` protocol.
    """
    obs_feats = context.feats[None, ...].astype(np.float32)        # (1, N, 2)
    obs_aids = context.asset_id[None, ...].astype(np.int64)        # (1, N)
    obs_tgts = context.iv[None, ...].astype(np.float32)            # (1, N)
    qry_feats = query.feats[None, ...].astype(np.float32)          # (1, Q, 2)
    qry_aids = query.asset_id[None, ...].astype(np.int64)          # (1, Q)
    return obs_feats, obs_aids, obs_tgts, qry_feats, qry_aids
