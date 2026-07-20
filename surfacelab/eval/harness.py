"""
The two evaluation harnesses every model shares.

run (independent / "perfect prior")
    For each eval day t: seed each model with *yesterday's full* quotes, then predict the
    full surface from a context subset.  Sweeps context sizes under two sampling regimes —
    `unif_{n}` (context sampled uniformly across the whole surface) and `extrap_{n}`
    (context sampled only inside the liquid box, so the model extrapolates to the wings /
    long maturities).  Each day is re-seeded with the true yesterday, so no error compounds.

run_sequential (decay test)
    Seed once on the first day, then walk forward calling `step`: tomorrow's prior is the
    model's *own* fit today.  Records per-day RMSE so error growth over time is visible.
    Sweeps several context sizes (each an independent walk) plus an extrapolation regime.

Both call only the SurfaceModel contract (seed_prior / predict / step), so they work for
every method unchanged.  Models must already be trained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from tqdm import tqdm

from surfacelab.core import arbitrage as arb
from surfacelab.eval.metrics import (
    split_stats, liquid_mask, call_price_stats,
)
from surfacelab.eval.records import Records
from surfacelab.eval.splitters import (
    Splitter, Full, LIQUID, _per_asset_sample, resolve_asset,
)

if TYPE_CHECKING:
    from surfacelab.core.model import SurfaceModel
    from surfacelab.data.dataset import Dataset

#: Offset added to the base seed for the "yesterday" sampling RNG, so the prior-seeding
#: stream is independent of today's context stream.
_PREV_STREAM_OFFSET = 9999


def _call_stats(dataset, t, idx, pred_iv, query):
    """Black-Scholes option-price diagnostics at the recorded query points (no-op when the
    dataset carries no bid/ask, e.g. a synthetic DGP).  `is_call` picks the call vs put
    price per point so an OTM surface (puts on k<0, calls on k>=0) is scored correctly."""
    bid = None if dataset.bid is None else dataset.bid[t, idx]
    ask = None if dataset.ask is None else dataset.ask[t, idx]
    is_call = None if dataset.is_call is None else dataset.is_call[t, idx]
    return call_price_stats(pred_iv, query.k, query.T, bid, ask, is_call)


def _prev_pool(dataset, d, regime, liquid) -> np.ndarray:
    """Day d's sampling pool for a regime: whole surface ('unif') or liquid box ('extrap').
    Used to seed a temporal model's prior from a *context-matched* sample of yesterday."""
    vp = dataset.valid_points(d)
    if regime == "extrap" and vp.size:
        return vp[liquid_mask(dataset.query_feats[d, vp, 0],
                              dataset.query_feats[d, vp, 1], **liquid)]
    return vp


def _no_arb(query, pred) -> tuple[float, float]:
    return (arb.butterfly_pct(query.k, query.T, pred, query.asset_id),
            arb.calendar_pct(query.k, query.T, pred, query.asset_id))


def run(models, dataset, ctx_sizes=(5, 10, 50, 100, 200, 300, 500), seed: int = 0,
        liquid: dict = LIQUID, eval_days=None, prior_ctx: str = "full") -> Records:
    """Independent evaluation with two context-sampling regimes.

    For each eval day t every model is seeded with yesterday's quotes, then predicts the
    *entire* surface (all valid points) from a context subset.  Two regimes, each swept
    over ``ctx_sizes`` (counts are per asset):

      * ``unif_{n}``   — n context points sampled uniformly across the whole surface.
      * ``extrap_{n}`` — n context points sampled uniformly but only from inside the liquid
                         box (|k| <= k_liq, T <= T_liq), so the model must extrapolate to
                         the unobserved wings / long maturities.

    ``prior_ctx`` controls how much of *yesterday* a temporal model is seeded with:
      * ``"full"``  — yesterday's complete surface (the "perfect prior" ceiling/reference).
      * ``"match"`` — yesterday sampled at the SAME nc and regime as today, so a model that
                      leans on yesterday (delta-CNP, bspline_temporal, prior, …) gets no
                      information the absolute models don't.  This is the fair regime.

    Targets are the full surface in *both* regimes, so the uniform curve is comparable
    across context sizes and, at the largest size (context ≈ whole surface), approaches an
    in-sample fit (RMSE -> ~0) — a sanity check on each fitter.
    """
    if prior_ctx not in ("full", "match"):
        raise ValueError(f"prior_ctx must be 'full' or 'match'; got {prior_ctx!r}")
    rec = Records()
    val = dataset.val_idx() if eval_days is None else np.asarray(eval_days)
    val = val[val > 0]                     # need a yesterday to seed the prior
    for model in models:
        rng = np.random.default_rng(seed)
        rng_prev = np.random.default_rng(seed + _PREV_STREAM_OFFSET)   # yesterday's sample: separate stream
        for t in tqdm(val, desc=f"{model.name} independent", leave=False):
            yest = dataset.quotes_at(t - 1)
            valid = dataset.valid_points(t)            # the full surface = target set
            if valid.size == 0:
                continue
            q_all = dataset.query_at(t, valid)
            true_all = dataset.targets[t, valid]
            # liquid-box pool the extrapolation regime samples its context from
            box = valid[liquid_mask(dataset.query_feats[t, valid, 0],
                                    dataset.query_feats[t, valid, 1], **liquid)]
            # yesterday's pools, for prior_ctx="match"
            valid_prev = dataset.valid_points(t - 1)
            box_prev = valid_prev[liquid_mask(dataset.query_feats[t - 1, valid_prev, 0],
                                              dataset.query_feats[t - 1, valid_prev, 1],
                                              **liquid)]

            for nc in ctx_sizes:
                for tag, pool in (("unif", valid), ("extrap", box)):
                    if prior_ctx == "match":
                        prev_pool = valid_prev if tag == "unif" else box_prev
                        pj = _per_asset_sample(dataset, t - 1, prev_pool, nc, rng_prev)
                        model.seed_prior(dataset.quotes_at(t - 1, pj))
                    else:
                        model.seed_prior(yest)
                    ci = _per_asset_sample(dataset, t, pool, nc, rng)
                    pred = model.predict(dataset.quotes_at(t, ci), q_all).iv
                    stats = split_stats(pred, true_all, q_all.k, q_all.T, **liquid)
                    stats.update(_call_stats(dataset, t, valid, pred, q_all))
                    bf, cal = _no_arb(q_all, pred)
                    rec.add(model.name, int(t), f"{tag}_{nc}", stats, bf, cal)
    return rec


def _walk(model, dataset, val, nc, regime, rng, liquid, rec,
          prior_ctx: str = "full", rng_prev=None):
    """One forward walk at context size `nc` under `regime` ∈ {"unif", "extrap"}.

    Predicts the full surface each day and carries the model's own fit forward; context is
    sampled uniformly across the whole surface ("unif") or only from inside the liquid box
    ("extrap").  Records the `{regime}_{nc}` split with per-day rows.

    ``prior_ctx`` controls the day-0 seed (and the per-step re-seed for perfect-persistence
    baselines): ``"full"`` uses yesterday's whole surface; ``"match"`` uses a sample of size
    `nc` from yesterday's same-regime pool, so a delta-CNP / bspline_temporal free-runs from a
    *realistic* prior instead of a perfect one (which otherwise flatters the whole walk)."""
    rng_prev = rng_prev if rng_prev is not None else np.random.default_rng(0)

    def _seed_from(day):
        if prior_ctx == "match":
            pj = _per_asset_sample(dataset, day, _prev_pool(dataset, day, regime, liquid),
                                   nc, rng_prev)
            model.seed_prior(dataset.quotes_at(day, pj))
        else:
            model.seed_prior(dataset.quotes_at(day))      # yesterday's full surface

    model.reset_sequence()
    _seed_from(int(val[0]) - 1)                            # day-0 prior
    for t in tqdm(val, desc=f"{model.name} {regime}_{nc}", leave=False):
        t = int(t)
        if model.reseed_each_step:        # baseline: re-seed on the true previous day
            _seed_from(t - 1)
        valid = dataset.valid_points(t)
        if valid.size == 0:
            continue
        q_all = dataset.query_at(t, valid)
        true_all = dataset.targets[t, valid]
        if regime == "extrap":
            pool = valid[liquid_mask(dataset.query_feats[t, valid, 0],
                                     dataset.query_feats[t, valid, 1], **liquid)]
        else:
            pool = valid
        ci = _per_asset_sample(dataset, t, pool, nc, rng)
        pred = model.step(dataset.quotes_at(t, ci), q_all).iv
        stats = split_stats(pred, true_all, q_all.k, q_all.T, **liquid)
        stats.update(_call_stats(dataset, t, valid, pred, q_all))
        bf, cal = _no_arb(q_all, pred)
        rec.add(model.name, t, f"{regime}_{nc}", stats, bf, cal)


def run_sequential(models, dataset, ctx_sizes=(5, 10, 50, 100, 200, 300, 500), seed: int = 0,
                   liquid: dict = LIQUID, eval_days=None, prior_ctx: str = "full") -> Records:
    """Sequential decay evaluation under the two sampling regimes, swept over context sizes.

    Each (model, regime, ctx_size) triple is an *independent* forward walk — seed once on
    day 0, then carry the model's own fit forward — because the carried prior depends on
    what context the model saw.  For every `ctx_sizes` entry there is a `unif_N` walk
    (context uniform over the whole surface) and an `extrap_N` walk (context only inside the
    liquid box).  Per-day rows are recorded so error growth over time is visible for each.
    """
    rec = Records()
    val = dataset.val_idx() if eval_days is None else np.asarray(eval_days)
    val = np.sort(val[val > 0])
    n_regimes = len(ctx_sizes) * 2
    pbar = tqdm(total=len(models) * n_regimes, desc="model×regime")
    for model in models:
        for nc in ctx_sizes:
            for regime in ("unif", "extrap"):
                _walk(model, dataset, val, nc, regime, np.random.default_rng(seed),
                      liquid, rec, prior_ctx=prior_ctx,
                      rng_prev=np.random.default_rng(seed + _PREV_STREAM_OFFSET))
                pbar.update(1)
    pbar.close()
    return rec


# ── leave-one-asset-out evaluation ───────────────────────────────────────────────
# Can a model reconstruct ONE asset's surface that it gets *no* context for, purely from
# the other assets' context (+ its carried prior)?  Each run scores that asset's full
# surface twice per regime — split `{regime}_N` (WITH its own context) and `{regime}_N_excl`
# (peers only) — so the gap isolates how much of the asset is recoverable cross-sectionally.

def _score_asset(rec, model, t, dataset, ex, pool, pred_all, split, liquid):
    """Record `split` for asset `ex` only, given predictions over the full `pool`."""
    m = dataset.asset_ids[t, pool] == ex
    if not m.any():
        return
    tgt_x = pool[m]
    q_x = dataset.query_at(t, tgt_x)
    pred, true_x = pred_all[m], dataset.targets[t, tgt_x]
    stats = split_stats(pred, true_x, q_x.k, q_x.T, **liquid)
    stats.update(_call_stats(dataset, t, tgt_x, pred, q_x))
    bf, cal = _no_arb(q_x, pred)
    rec.add(model.name, int(t), split, stats, bf, cal)


def run_exclude(models, dataset, exclude_asset, ctx_sizes=(5, 10, 50, 100, 200, 300, 500),
                seed: int = 0, liquid: dict = LIQUID, eval_days=None,
                prior_ctx: str = "full") -> Records:
    """Independent leave-one-asset-out: each day predict `exclude_asset`'s full surface WITH
    its own context (`{regime}_N`) and WITHOUT it (`{regime}_N_excl`, only the other assets
    observed), under both sampling regimes (`unif` / `extrap`).  Peers' context is identical
    between the two, so the gap is purely the value of the asset's own quotes vs what the
    model infers cross-sectionally.

    ``prior_ctx`` ("full" | "match") seeds yesterday's prior with the whole surface or a
    context-matched sparse sample, exactly as in `run` (the seed is identical across the
    with/without-context tags, so only today's context differs between them)."""
    ex = resolve_asset(dataset, exclude_asset)
    rec = Records()
    val = dataset.val_idx() if eval_days is None else np.asarray(eval_days)
    val = val[val > 0]
    for model in models:
        rng = np.random.default_rng(seed)
        rng_prev = np.random.default_rng(seed + _PREV_STREAM_OFFSET)
        for t in tqdm(val, desc=f"{model.name} excl", leave=False):
            t = int(t)
            yest = dataset.quotes_at(t - 1)
            valid = dataset.valid_points(t)
            if valid.size == 0 or not (dataset.asset_ids[t, valid] == ex).any():
                continue
            q_all = dataset.query_at(t, valid)
            box = valid[liquid_mask(dataset.query_feats[t, valid, 0],
                                    dataset.query_feats[t, valid, 1], **liquid)]
            for nc in ctx_sizes:
                for regime, pool in (("unif", valid), ("extrap", box)):
                    ci_full = _per_asset_sample(dataset, t, pool, nc, rng)
                    ci_excl = ci_full[dataset.asset_ids[t, ci_full] != ex]
                    if prior_ctx == "match":
                        pj = _per_asset_sample(dataset, t - 1,
                                               _prev_pool(dataset, t - 1, regime, liquid),
                                               nc, rng_prev)
                        seed_quotes = dataset.quotes_at(t - 1, pj)
                    else:
                        seed_quotes = yest
                    for tag, ci in (("", ci_full), ("_excl", ci_excl)):
                        model.seed_prior(seed_quotes)
                        pred_all = model.predict(dataset.quotes_at(t, ci), q_all).iv
                        _score_asset(rec, model, t, dataset, ex, valid, pred_all,
                                     f"{regime}_{nc}{tag}", liquid)
    return rec


def _walk_excl(model, dataset, val, nc, regime, rng, ex, exclude: bool, rec, split, liquid,
               prior_ctx: str = "full", rng_prev=None):
    """One forward walk scoring asset `ex` only; if `exclude`, that asset is never in the
    context (predicted purely from peers + its carried prior).  Context is sampled uniformly
    over the surface ("unif") or only inside the liquid box ("extrap").  ``prior_ctx`` seeds
    the day-0 (and per-step baseline) prior full vs context-matched sparse, as in `_walk`."""
    rng_prev = rng_prev if rng_prev is not None else np.random.default_rng(0)

    def _seed_from(day):
        if prior_ctx == "match":
            pj = _per_asset_sample(dataset, day, _prev_pool(dataset, day, regime, liquid),
                                   nc, rng_prev)
            model.seed_prior(dataset.quotes_at(day, pj))
        else:
            model.seed_prior(dataset.quotes_at(day))

    model.reset_sequence()
    _seed_from(int(val[0]) - 1)
    for t in tqdm(val, desc=f"{model.name} {split}", leave=False):
        t = int(t)
        if model.reseed_each_step:
            _seed_from(t - 1)
        valid = dataset.valid_points(t)
        if valid.size == 0:
            continue
        if regime == "extrap":
            pool = valid[liquid_mask(dataset.query_feats[t, valid, 0],
                                     dataset.query_feats[t, valid, 1], **liquid)]
        else:
            pool = valid
        ci = _per_asset_sample(dataset, t, pool, nc, rng)
        if exclude:
            ci = ci[dataset.asset_ids[t, ci] != ex]
        # predict the FULL surface so the carried prior stays well-maintained for every
        # asset; only asset `ex` is scored.
        pred_all = model.step(dataset.quotes_at(t, ci), dataset.query_at(t, valid)).iv
        _score_asset(rec, model, t, dataset, ex, valid, pred_all, split, liquid)


# ── asymmetric-liquidity evaluation ──────────────────────────────────────────────
# The realistic regime the symmetric harness hides: every OTHER asset is fully observed
# (the liquid market), while the TARGET asset gets only `nc` of its own quotes.  We sweep
# the TARGET's quote count and score the target only.  A cross-asset model can anchor the
# target's level on its 1-2 quotes and borrow the day's MOVE from the fully-observed peers,
# so reconstruction should already be good at nc = 2-3 — exactly the asymmetry that makes
# cross-asset coupling worth something (see surfacelab/statistics).

def run_target_asymmetric(models, dataset, target_asset,
                          ctx_sizes=(1, 2, 3, 5, 10, 20, 50, 100),
                          seed: int = 0, liquid: dict = LIQUID, eval_days=None,
                          prior_ctx: str = "full") -> Records:
    """Independent asymmetric-liquidity reconstruction of ONE target asset.

    Peers always get their FULL context (all valid quotes that day); the target gets `nc` of
    its own quotes, sampled uniformly over its surface (`unif_N`) or only inside the liquid
    box (`extrap_N`).  Only the target's surface is scored, so every split is target-only and
    the x-axis is the TARGET's own quote count.  Peers' context is identical across nc, so the
    curve isolates how few target quotes are needed once the rest of the market is liquid.

    prior_ctx: "full" seeds yesterday's whole surface (the natural strong prior — we always
    have yesterday); "match" seeds a sparse yesterday sample so the carried prior is weak and
    today's information (the target's nc quotes + the liquid peers) does the work."""
    ex = resolve_asset(dataset, target_asset)
    rec = Records()
    val = dataset.val_idx() if eval_days is None else np.asarray(eval_days)
    val = val[val > 0]
    for model in models:
        rng = np.random.default_rng(seed)
        rng_prev = np.random.default_rng(seed + _PREV_STREAM_OFFSET)
        for t in tqdm(val, desc=f"{model.name} asym", leave=False):
            t = int(t)
            valid = dataset.valid_points(t)
            if valid.size == 0 or not (dataset.asset_ids[t, valid] == ex).any():
                continue
            q_all = dataset.query_at(t, valid)
            peer_full = valid[dataset.asset_ids[t, valid] != ex]   # ALL peer quotes today (liquid)
            box = valid[liquid_mask(dataset.query_feats[t, valid, 0],
                                    dataset.query_feats[t, valid, 1], **liquid)]
            # yesterday: peers always fully observed (they are liquid every day).  Under
            # prior_ctx="match" the TARGET's prior is sparse too (its nc quotes from yesterday),
            # so the carried AAPL prior is realistically weak and today's information — its few
            # quotes + the liquid peers — must do the work.  Under "full" the whole surface seeds.
            yest_valid = dataset.valid_points(t - 1)
            yest_peers = yest_valid[dataset.asset_ids[t - 1, yest_valid] != ex]
            for nc in ctx_sizes:
                for regime, pool in (("unif", valid), ("extrap", box)):
                    tgt_pool = pool[dataset.asset_ids[t, pool] == ex]
                    if len(tgt_pool):
                        ti = rng.choice(tgt_pool, min(nc, len(tgt_pool)), replace=False)
                        ci = np.concatenate([peer_full, ti])
                    else:
                        ci = peer_full
                    if prior_ctx == "match":
                        ypool = _prev_pool(dataset, t - 1, regime, liquid)
                        ytgt = ypool[dataset.asset_ids[t - 1, ypool] == ex]
                        yti = (rng_prev.choice(ytgt, min(nc, len(ytgt)), replace=False)
                               if len(ytgt) else np.empty(0, int))
                        seed_quotes = dataset.quotes_at(t - 1, np.concatenate([yest_peers, yti]))
                    else:
                        seed_quotes = dataset.quotes_at(t - 1)
                    model.seed_prior(seed_quotes)
                    pred_all = model.predict(dataset.quotes_at(t, ci), q_all).iv
                    _score_asset(rec, model, t, dataset, ex, valid, pred_all,
                                 f"{regime}_{nc}", liquid)
    return rec


def run_sequential_exclude(models, dataset, exclude_asset,
                           ctx_sizes=(5, 10, 50, 100, 200, 300, 500),
                           seed: int = 0, liquid: dict = LIQUID, eval_days=None,
                           prior_ctx: str = "full") -> Records:
    """Sequential leave-one-asset-out: per context size and sampling regime, two independent
    forward walks — one where `exclude_asset` keeps its own context (`{regime}_N`), one where
    it never gets context (`{regime}_N_excl`).  Both free-run; the excluded asset's surface
    is carried forward from the model's own peer-driven predictions.  The same seed is used
    for both walks, so peers see identical context."""
    ex = resolve_asset(dataset, exclude_asset)
    rec = Records()
    val = dataset.val_idx() if eval_days is None else np.asarray(eval_days)
    val = np.sort(val[val > 0])
    pbar = tqdm(total=len(models) * len(ctx_sizes) * 4, desc="excl model×regime")
    for model in models:
        for nc in ctx_sizes:
            for regime in ("unif", "extrap"):
                _walk_excl(model, dataset, val, nc, regime, np.random.default_rng(seed),
                           ex, False, rec, f"{regime}_{nc}", liquid,
                           prior_ctx=prior_ctx, rng_prev=np.random.default_rng(seed + _PREV_STREAM_OFFSET))
                pbar.update(1)
                _walk_excl(model, dataset, val, nc, regime, np.random.default_rng(seed),
                           ex, True, rec, f"{regime}_{nc}_excl", liquid,
                           prior_ctx=prior_ctx, rng_prev=np.random.default_rng(seed + _PREV_STREAM_OFFSET))
                pbar.update(1)
    pbar.close()
    return rec


# ══════════════════════════════════════════════════════════════════════════════════
# Unified, composable harness: one loop driven by a Model bundle.
#
# A `Model` bundles the four pieces an experiment cares about — a (trained) fitter, a
# *today* splitter, a *yesterday* splitter, and how the prior is carried — so the four
# `run_*` functions above collapse into the single `run_models` loop below.  The legacy
# functions are kept so existing configs keep working; new configs use `run_models`.
# ══════════════════════════════════════════════════════════════════════════════════
@dataclass
class Model:
    """One evaluation unit: a fitter plus how it sees today and yesterday.

      * ``fitter``     — a *built and trained* SurfaceModel (the "today fitting method").
      * ``today``      — splitter for today's context + the points to score.
      * ``yesterday``  — splitter whose reveal seeds the prior (the "yesterday fitting
                         method"); defaults to the full surface (perfect prior).
      * ``prior_mode`` — ``"fit"`` re-derives the prior from ``yesterday`` each day
                         (independent); ``"carry"`` seeds once then rolls the model's own
                         fit forward via ``step`` (sequential).
      * ``dataset``    — optional per-model dataset override; ``None`` → the experiment's.
      * ``reseed_each_step`` — in ``carry`` mode, re-seed on the true previous day every
                         step (perfect-persistence baseline).

    Rows are recorded under ``name`` (defaults to the fitter's name) and ``today.label``,
    so several Models sharing a fitter but differing in splitter land under one model name
    with distinct splits — exactly the old ``{regime}_{nc}`` layout.
    """
    fitter: "SurfaceModel"
    today: Splitter
    yesterday: Splitter = field(default_factory=Full)
    prior_mode: str = "fit"
    dataset: Optional["Dataset"] = None
    reseed_each_step: bool = False
    name: Optional[str] = None

    def __post_init__(self):
        if self.prior_mode not in ("fit", "carry"):
            raise ValueError(f"prior_mode must be 'fit' or 'carry'; got {self.prior_mode!r}")
        if self.name is None:
            self.name = self.fitter.name


def _score(rec, m, t, dataset, valid, score_idx, pred_full, liquid):
    """Record one row: score `score_idx` (⊆ valid) given a full-surface prediction."""
    if score_idx.size == 0:
        return
    # positions of score_idx within valid (pred_full is aligned to valid)
    pos = np.searchsorted(valid, score_idx)
    pred = pred_full[pos]
    q = dataset.query_at(t, score_idx)
    true = dataset.targets[t, score_idx]
    stats = split_stats(pred, true, q.k, q.T, **liquid)
    stats.update(_call_stats(dataset, t, score_idx, pred, q))
    bf, cal = _no_arb(q, pred)
    rec.add(m.name, int(t), m.today.label, stats, bf, cal)


def run_models(models, dataset, *, seed: int = 0, liquid: dict = LIQUID,
               eval_days=None) -> Records:
    """Run a list of `Model` bundles through one generic day loop.

    For every model and every eval day: seed the prior (re-fit from ``yesterday`` in
    ``fit`` mode, or carry the model's own fit in ``carry`` mode), reveal today's context,
    predict the **full** surface (so the carried prior stays well maintained for every
    asset), then score only ``today``'s ``score_idx``.  Reproduces the methodology of the
    legacy harness; absolute numbers may differ slightly because sampling is now seeded
    per model rather than shared across the whole sweep.
    """
    rec = Records()
    for m in models:
        ds = m.dataset or dataset
        val = ds.val_idx() if eval_days is None else np.asarray(eval_days)
        val = np.sort(val[val > 0])
        if val.size == 0:
            continue
        # per-model streams so each model/split is independent and reproducible
        key = abs(hash(m.name + m.today.label)) % (2 ** 31)
        rng = np.random.default_rng(seed + key)
        rng_prev = np.random.default_rng(seed + _PREV_STREAM_OFFSET + key)

        def seed_from(day):
            reveal, _ = m.yesterday(ds, int(day), rng_prev)
            m.fitter.seed_prior(ds.quotes_at(int(day), reveal))

        # CARRY MODE: the `yesterday` fitter/splitter is used ONLY ONCE, here, to seed the
        # day-0 prior.  From then on the prior IS the model's own fit, rolled forward by
        # step() below — yesterday is never consulted again.  (The sole exception is the
        # perfect-persistence baseline `reseed_each_step`, which deliberately re-seeds from
        # the true previous day every step.)  In FIT mode, by contrast, `yesterday` is
        # re-applied every day inside the loop.
        if m.prior_mode == "carry":
            m.fitter.reset_sequence()
            seed_from(val[0] - 1)

        for t in tqdm(val, desc=f"{m.name} {m.today.label}", leave=False):
            t = int(t)
            # re-seed from yesterday only when re-fitting each day (FIT), or for the
            # reseed-each-step baseline; plain CARRY skips this and relies on step().
            if m.prior_mode == "fit" or (m.prior_mode == "carry" and m.reseed_each_step):
                seed_from(t - 1)
            valid = ds.valid_points(t)
            if valid.size == 0:
                continue
            reveal, score_idx = m.today(ds, t, rng)
            q_full = ds.query_at(t, valid)
            ctx = ds.quotes_at(t, reveal)
            fit_fn = m.fitter.step if m.prior_mode == "carry" else m.fitter.predict
            pred_full = fit_fn(ctx, q_full).iv
            _score(rec, m, t, ds, valid, score_idx, pred_full, liquid)
    return rec
