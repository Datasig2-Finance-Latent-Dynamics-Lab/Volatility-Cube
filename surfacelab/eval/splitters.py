"""Composable context/target splitters.

The sampling logic that used to be inlined across the four ``run_*`` harness functions,
pulled out into small objects so an experiment can *compose* what it reveals instead of
needing a bespoke function per regime.

A ``Splitter``, given ``(dataset, day t, rng)``, returns ``(reveal_idx, score_idx)``:

  * ``reveal_idx`` — point indices revealed to the fitter as context that day.
  * ``score_idx``  — point indices to predict and score (the targets); always a subset of
    the day's valid points.

The model's *today* splitter uses both; its *yesterday* splitter uses only ``reveal_idx``
(to seed/fit the prior).  ``label`` is the split name recorded for each row (``extrap_8``…),
so it plays the role the old ``f"{regime}_{nc}"`` tag did.
"""
from __future__ import annotations

import numpy as np

from surfacelab.eval.metrics import liquid_mask

LIQUID = dict(k_liq=0.2, T_liq=0.5)


# ── shared sampling helpers (moved verbatim from harness) ─────────────────────────
def _box(ds, t, idx, liquid) -> np.ndarray:
    """Restrict point indices `idx` (day t) to the liquid box."""
    if idx.size == 0:
        return idx
    return idx[liquid_mask(ds.query_feats[t, idx, 0], ds.query_feats[t, idx, 1], **liquid)]


def _per_asset_sample(ds, t, pool, n, rng) -> np.ndarray:
    """Sample up to `n` context indices per asset from `pool` (day t)."""
    aid = ds.asset_ids[t]
    out = []
    for a in np.unique(aid[pool]):
        pa = pool[aid[pool] == a]
        if len(pa):
            out.append(rng.choice(pa, min(n, len(pa)), replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def resolve_asset(ds, asset) -> int:
    """Asset name (or id) → 0-based id."""
    names = list(ds.meta.get("asset_names", []))
    if isinstance(asset, str):
        if asset not in names:
            raise ValueError(f"asset {asset!r} not in asset_names {names}")
        return names.index(asset)
    return int(asset)


# ── splitters ─────────────────────────────────────────────────────────────────────
class Splitter:
    """Maps (dataset, day, rng) → (reveal_idx, score_idx).  Subclasses set ``label``."""
    label: str = "split"

    def __call__(self, ds, t, rng) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class Full(Splitter):
    """Reveal (and score) the whole surface.  The perfect-prior 'yesterday' splitter, and
    the in-sample reference 'today' splitter."""
    label = "full"

    def __call__(self, ds, t, rng):
        v = ds.valid_points(t)
        return v, v


class Uniform(Splitter):
    """`n` context points per asset sampled across the whole surface; score the full surface.
    The old ``unif_{n}`` regime."""

    def __init__(self, n: int):
        self.n = n
        self.label = f"unif_{n}"

    def __call__(self, ds, t, rng):
        v = ds.valid_points(t)
        return _per_asset_sample(ds, t, v, self.n, rng), v


class Extrap(Splitter):
    """`n` context points per asset from inside the liquid box; score the full surface, so the
    model extrapolates to the wings.  The old ``extrap_{n}`` regime."""

    def __init__(self, n: int, liquid: dict = LIQUID):
        self.n = n
        self.liquid = liquid
        self.label = f"extrap_{n}"

    def __call__(self, ds, t, rng):
        v = ds.valid_points(t)
        return _per_asset_sample(ds, t, _box(ds, t, v, self.liquid), self.n, rng), v


class Matched(Splitter):
    """A 'yesterday' splitter: reveal `n` per asset from the same regime's pool — the old
    ``prior_ctx='match'`` behaviour (the prior sees no more than today does)."""

    def __init__(self, n: int, regime: str = "extrap", liquid: dict = LIQUID):
        self.n = n
        self.regime = regime
        self.liquid = liquid
        self.label = f"matched_{regime}_{n}"

    def __call__(self, ds, t, rng):
        v = ds.valid_points(t)
        pool = _box(ds, t, v, self.liquid) if self.regime == "extrap" else v
        return _per_asset_sample(ds, t, pool, self.n, rng), v


class Asymmetric(Splitter):
    """Asymmetric liquidity: peers fully revealed, the target gets `n` of its own quotes (from
    the regime pool); score the target only.  Replaces ``run_target_asymmetric``."""

    def __init__(self, target, n: int, regime: str = "extrap", liquid: dict = LIQUID):
        self.target = target
        self.n = n
        self.regime = regime
        self.liquid = liquid
        self.label = f"{regime}_{n}"

    def __call__(self, ds, t, rng):
        v = ds.valid_points(t)
        ex = resolve_asset(ds, self.target)
        peers = v[ds.asset_ids[t, v] != ex]
        pool = _box(ds, t, v, self.liquid) if self.regime == "extrap" else v
        tgt_pool = pool[ds.asset_ids[t, pool] == ex]
        if len(tgt_pool):
            ti = rng.choice(tgt_pool, min(self.n, len(tgt_pool)), replace=False)
            reveal = np.concatenate([peers, ti])
        else:
            reveal = peers
        score = v[ds.asset_ids[t, v] == ex]
        return reveal, score


class Exclude(Splitter):
    """Leave-one-asset-out: peers revealed, the target gets NO context; score the target only.
    Replaces the ``_excl`` arm of ``run_exclude``."""

    def __init__(self, target, n: int, regime: str = "extrap", liquid: dict = LIQUID):
        self.target = target
        self.n = n
        self.regime = regime
        self.liquid = liquid
        self.label = f"{regime}_{n}_excl"

    def __call__(self, ds, t, rng):
        v = ds.valid_points(t)
        ex = resolve_asset(ds, self.target)
        pool = _box(ds, t, v, self.liquid) if self.regime == "extrap" else v
        # peers sampled at n/asset (same as the with-context arm), target dropped entirely
        ci = _per_asset_sample(ds, t, pool, self.n, rng)
        reveal = ci[ds.asset_ids[t, ci] != ex]
        score = v[ds.asset_ids[t, v] == ex]
        return reveal, score
