"""
The canonical dataset container + bridges to the model contract.

`Dataset` carries the same padded per-day arrays the project has always used
(query_feats, asset_ids, targets, ...), and adds helpers that cut one day into the
`Quotes` / `QueryPoints` the harness feeds to models.  Padding rows (T == 0) are
treated as absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from surfacelab.core.types import Quotes, QueryPoints


@dataclass
class Dataset:
    """Padded per-day surface dataset.

    query_feats : (N_days, N_points, Q_dim)  — Q_dim = 2 → (lm, T)
    asset_ids   : (N_days, N_points)  int, 0-based
    targets     : (N_days, N_points)  float — IV
    split       : (N_days,)  int8 — 0 train / 1 val
    ctx_max     : first ctx_max points per day are context candidates
    params      : (N_days, N_assets, P_dim) optional ground-truth params
    bid, ask    : (N_days, N_points) optional forward-normalised quotes
    is_call     : (N_days, N_points) optional bool — True if the quote at this point is a
                  call, False if a put.  Needed so the call/put-price diagnostic compares
                  the predicted IV against the matching option price (OTM mixes both).
    prior_targets : (N_days, N_points) optional — yesterday's surface here (NaN if none)
    """

    query_feats: np.ndarray
    asset_ids: np.ndarray
    targets: np.ndarray
    split: np.ndarray
    ctx_max: int
    n_assets: int
    params: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)
    bid: Optional[np.ndarray] = None
    ask: Optional[np.ndarray] = None
    is_call: Optional[np.ndarray] = None
    prior_targets: Optional[np.ndarray] = None

    # ── shape helpers ─────────────────────────────────────────────────────────
    @property
    def n_days(self) -> int:
        return self.query_feats.shape[0]

    @property
    def n_points(self) -> int:
        return self.query_feats.shape[1]

    @property
    def q_dim(self) -> int:
        return self.query_feats.shape[2]

    def train_idx(self) -> np.ndarray:
        return np.where(self.split == 0)[0]

    def val_idx(self) -> np.ndarray:
        return np.where(self.split == 1)[0]

    def subset(self, indices: np.ndarray) -> "Dataset":
        return Dataset(
            query_feats=self.query_feats[indices],
            asset_ids=self.asset_ids[indices],
            targets=self.targets[indices],
            split=self.split[indices],
            ctx_max=self.ctx_max,
            n_assets=self.n_assets,
            params=self.params[indices] if self.params is not None else None,
            meta=self.meta.copy(),
            bid=self.bid[indices] if self.bid is not None else None,
            ask=self.ask[indices] if self.ask is not None else None,
            is_call=self.is_call[indices] if self.is_call is not None else None,
            prior_targets=(self.prior_targets[indices]
                           if self.prior_targets is not None else None),
        )

    # ── bridges to the model contract ─────────────────────────────────────────
    def valid_mask(self, t: int) -> np.ndarray:
        """Bool mask over points of day t that are real (not zero-padding)."""
        return self.query_feats[t, :, 1] > 0

    def valid_points(self, t: int) -> np.ndarray:
        """Integer indices of the real points on day t."""
        return np.where(self.valid_mask(t))[0]

    def context_pool(self, t: int) -> np.ndarray:
        """Real point indices among the first ctx_max positions (context candidates)."""
        return self.valid_points(t)[self.valid_points(t) < self.ctx_max]

    def target_pool(self, t: int) -> np.ndarray:
        """Real point indices at positions >= ctx_max (held-out target candidates)."""
        return self.valid_points(t)[self.valid_points(t) >= self.ctx_max]

    def quotes_at(self, t: int, idx: np.ndarray | None = None) -> Quotes:
        """Build Quotes from a set of point indices on day t (default: all real points)."""
        if idx is None:
            idx = self.valid_points(t)
        idx = np.asarray(idx, dtype=int)
        qf = self.query_feats[t, idx]
        return Quotes(
            k=qf[:, 0], T=qf[:, 1],
            iv=self.targets[t, idx],
            asset_id=self.asset_ids[t, idx],
            bid=None if self.bid is None else self.bid[t, idx],
            ask=None if self.ask is None else self.ask[t, idx],
        )

    def query_at(self, t: int, idx: np.ndarray | None = None,
                 use_prior: bool = False) -> QueryPoints:
        """Build QueryPoints from point indices on day t (default: all real points).

        If use_prior, attach prior_targets[t, idx] as ``prior_iv`` (for delta models).
        """
        if idx is None:
            idx = self.valid_points(t)
        idx = np.asarray(idx, dtype=int)
        qf = self.query_feats[t, idx]
        prior_iv = None
        if use_prior and self.prior_targets is not None:
            prior_iv = self.prior_targets[t, idx]
        return QueryPoints(
            k=qf[:, 0], T=qf[:, 1],
            asset_id=self.asset_ids[t, idx],
            prior_iv=prior_iv,
        )


# Backward-compatible alias for code/ports that reference the old name.
SurfaceDataset = Dataset
