"""
Model-agnostic plots driven by the SurfaceModel contract and Records.

  * plot_reconstruction   — one day, per asset: true (dashed) vs predicted (solid) smiles.
  * plot_rmse_vs_ctx      — RMSE vs context size, one line per model (from a summary).
  * plot_rmse_vs_ctx_lastday — RMSE vs context size on the last sequential day.
  * plot_rmse_decay       — per-day RMSE over a sequential run (error growth over time).
  * plot_spread_vs_ctx / _lastday / plot_spread_decay — companions of the three RMSE curves
    above, but on the y-axis the spread-based "miss": the mean distance the prediction falls
    outside the bid-ask spread, in spread-widths, over the out-of-spread points (the
    `call_oob_spread_mean` from the summary / `call_oob_spread_sum / call_oob` per record).
    A miss below 1 spread-width is, in practice, a usable quote; the dashed line marks it.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# RMSE plots are bounded to roughly 1–10 vol points (IV units): that's the band that matters
# for the dissertation.  Models below ~0.5 vol pt sit at the floor; diverged ones (>10 vol pt)
# run off the top rather than crushing everyone else's scale.
RMSE_YLIM = (0.005, 0.10)

# Spread-miss plots are in spread-widths (dimensionless): 0 is on top of mid, 1 is one full
# spread out.  The interesting band runs from the floor up to ~20 widths for the worst
# sparse fits, so a linear axis from 0 reads naturally; the dashed line marks one spread.
SPREAD_YLIM = (0.0, 20.0)
SPREAD_YLABEL = "miss (spread-widths, out-of-spread pts)"


def _save(fig, out_path):
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def split_nc(split: str, prefix: str):
    """Context size from a `<prefix><int>` split (e.g. split_nc('extrap_20', 'extrap_') == 20),
    or None if it doesn't match that regime prefix.  A trailing '_excl' (leave-one-out splits
    like 'unif_10_excl') is stripped first so cold-start runs plot too."""
    if not split.startswith(prefix):
        return None
    tail = split[len(prefix):]
    if tail.endswith("_excl"):
        tail = tail[:-5]
    return int(tail) if tail.isdigit() else None


def plot_reconstruction(model, dataset, day: int, n_ctx: int = 50,
                        seed: int = 0, out_path: str | None = None,
                        only_asset: int | None = None, sparse_asset: int | None = None,
                        n_ctx_sparse: int = 3):
    """Per-asset reconstruction on one day: dashed = true, solid = predicted.

    `only_asset`   — plot just this asset's panel (the rest are still used as context).
    `sparse_asset` — asymmetric regime: this asset gets only `n_ctx_sparse` quotes while
                     every OTHER asset gets its FULL context (used to visualise predicting a
                     sparse target from fully-observed peers)."""
    rng = np.random.default_rng(seed)
    model.reset_sequence()                       # clear any state carried from a prior call
    if day > 0:
        model.seed_prior(dataset.quotes_at(day - 1))
    cpool, tpool = dataset.context_pool(day), dataset.target_pool(day)
    aid = dataset.asset_ids[day]
    ci = []
    for a in np.unique(aid[cpool]):
        pa = cpool[aid[cpool] == a]
        if sparse_asset is not None:
            k = n_ctx_sparse if a == sparse_asset else len(pa)   # target sparse, peers full
        else:
            k = n_ctx
        ci.append(rng.choice(pa, min(k, len(pa)), replace=False))
    ctx = dataset.quotes_at(day, np.concatenate(ci))
    q = dataset.query_at(day, tpool)
    pred = model.predict(ctx, q).iv
    true = dataset.targets[day, tpool]

    assets = np.unique(q.asset_id)
    if only_asset is not None:
        assets = assets[assets == only_asset]
    names = dataset.meta.get("asset_names", [f"asset {a}" for a in range(dataset.n_assets)])
    fig, axes = plt.subplots(1, len(assets), figsize=(4 * len(assets), 3.2), squeeze=False)
    for ax, a in zip(axes[0], assets):
        m = q.asset_id == a
        Tk = np.round(q.T[m], 4)
        for tv in np.unique(Tk):
            s = Tk == tv
            o = np.argsort(q.k[m][s])
            c = plt.cm.viridis(tv / max(Tk.max(), 1e-6))
            ax.plot(q.k[m][s][o], true[m][s][o], "--", color=c, lw=1.0, alpha=0.6)
            ax.plot(q.k[m][s][o], pred[m][s][o], "-", color=c, lw=1.4)
        ax.set_title(names[int(a)], fontsize=9)
        ax.set_xlabel("log-moneyness"); ax.tick_params(labelsize=7)
    fig.suptitle(f"{model.name}  day {day}  (n_ctx={n_ctx}, dashed=true)", fontsize=11)
    _save(fig, out_path)
    return fig


def plot_rmse_vs_ctx(summary_rows, out_path: str | None = None,
                     title: str = "RMSE vs context size", prefix: str = "unif_"):
    """summary_rows: Records.summary() output. Plots <prefix>N splits per model.

    `prefix` selects the sampling regime ("unif_" or "extrap_"); legacy "ctx_" still works.
    """
    by_model: dict = {}
    for r in summary_rows:
        nc = split_nc(r["split"], prefix)
        if nc is not None:
            by_model.setdefault(r["model"], []).append((nc, r["rmse"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", label=model, lw=1.5, ms=4)
    ax.set_xlabel("context size"); ax.set_ylabel("RMSE (IV)")
    ax.set_yscale("log"); ax.set_ylim(*RMSE_YLIM)
    ax.set_title(title); ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, out_path)
    return fig


def plot_rmse_vs_ctx_lastday(records, out_path: str | None = None,
                             title: str = "RMSE vs context size (last day)",
                             prefix: str = "unif_"):
    """From sequential records, plot RMSE vs context size on the *last* eval day,
    one line per model — how much extra context helps once state has propagated.

    `prefix` selects the sampling regime ("unif_" / "extrap_"; legacy "ctx_" still works)."""
    def _nc(split):
        return split_nc(split, prefix)
    last_day = max((r["day"] for r in records.rows if _nc(r["split"]) is not None),
                   default=None)
    if last_day is None:
        return None
    by_model: dict = {}
    for r in records.rows:
        nc = _nc(r["split"])
        if r["day"] != last_day or nc is None or not r["n"]:
            continue
        by_model.setdefault(r["model"], []).append((nc, np.sqrt(r["sq"] / r["n"])))
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", label=model, lw=1.5, ms=4)
    ax.set_xlabel("context size"); ax.set_ylabel("RMSE (IV)")
    ax.set_yscale("log"); ax.set_ylim(*RMSE_YLIM)
    ax.set_title(f"{title} · day {last_day}")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    _save(fig, out_path)
    return fig


def plot_rmse_decay(records, split: str = "unif_100",
                    out_path: str | None = None,
                    title: str = "Sequential RMSE over time"):
    """Per-day RMSE for each model over a sequential run (rows tagged by day)."""
    by_model: dict = {}
    for r in records.rows:
        if r["split"] != split:
            continue
        rmse = np.sqrt(r["sq"] / r["n"]) if r["n"] else np.nan
        by_model.setdefault(r["model"], []).append((r["day"], rmse))
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(range(len(xs)), ys, "-", label=model, lw=1.4)
    ax.set_xlabel("days into sequence"); ax.set_ylabel(f"RMSE ({split})")
    ax.set_ylim(*RMSE_YLIM)
    ax.set_title(title); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    _save(fig, out_path)
    return fig


# ── spread-miss companions of the three RMSE curves ─────────────────────────────
def _row_spread_miss(r):
    """Per-record mean miss in spread-widths (call_oob_spread_sum / call_oob), or NaN if no
    point fell outside the spread that day (nothing to average)."""
    oob = r.get("call_oob", 0)
    return (r["call_oob_spread_sum"] / oob) if oob else np.nan


def _finish_spread_ax(ax, title):
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8)        # one spread-width: usable-quote line
    ax.set_ylabel(SPREAD_YLABEL); ax.set_ylim(*SPREAD_YLIM)
    ax.set_title(title); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)


def plot_spread_vs_ctx(summary_rows, out_path: str | None = None,
                       title: str = "Spread miss vs context size", prefix: str = "unif_"):
    """Companion of plot_rmse_vs_ctx: pooled spread miss (`call_oob_spread_mean`) vs context."""
    by_model: dict = {}
    for r in summary_rows:
        nc = split_nc(r["split"], prefix)
        if nc is not None and np.isfinite(r.get("call_oob_spread_mean", np.nan)):
            by_model.setdefault(r["model"], []).append((nc, r["call_oob_spread_mean"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", label=model, lw=1.5, ms=4)
    ax.set_xlabel("context size")
    _finish_spread_ax(ax, title)
    _save(fig, out_path)
    return fig


def plot_spread_vs_ctx_lastday(records, out_path: str | None = None,
                               title: str = "Spread miss vs context size (last day)",
                               prefix: str = "unif_"):
    """Companion of plot_rmse_vs_ctx_lastday: per-model spread miss on the last eval day."""
    def _nc(split):
        return split_nc(split, prefix)
    last_day = max((r["day"] for r in records.rows if _nc(r["split"]) is not None),
                   default=None)
    if last_day is None:
        return None
    by_model: dict = {}
    for r in records.rows:
        nc = _nc(r["split"])
        if r["day"] != last_day or nc is None:
            continue
        miss = _row_spread_miss(r)
        if np.isfinite(miss):
            by_model.setdefault(r["model"], []).append((nc, miss))
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", label=model, lw=1.5, ms=4)
    ax.set_xlabel("context size")
    _finish_spread_ax(ax, f"{title} · day {last_day}")
    _save(fig, out_path)
    return fig


def plot_spread_decay(records, split: str = "unif_100",
                      out_path: str | None = None,
                      title: str = "Sequential spread miss over time"):
    """Companion of plot_rmse_decay: per-day spread miss for each model over a sequential run."""
    by_model: dict = {}
    for r in records.rows:
        if r["split"] != split:
            continue
        by_model.setdefault(r["model"], []).append((r["day"], _row_spread_miss(r)))
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(range(len(xs)), ys, "-", label=model, lw=1.4)
    ax.set_xlabel("days into sequence")
    _finish_spread_ax(ax, title)
    _save(fig, out_path)
    return fig
