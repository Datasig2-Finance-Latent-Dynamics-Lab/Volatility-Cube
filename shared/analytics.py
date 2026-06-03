"""
Array-based analytics that work with any SurfacePredictor.

These functions accept plain numpy arrays rather than framework-specific
dataset objects so that both neural_processes and dgraph experiments can
use them for cross-system comparison.

Usage (neural_processes):
    from shared.analytics import eval_rmse, plot_reconstruction
    rmse = eval_rmse(fitted_cnp, qry_feats, asset_ids, targets, ctx_max, [5,20,50])

Usage (dgraph vol_surface):
    from dgraph.examples.vol_surface.source.predictor import DGraphSurfacePredictor
    from shared.analytics import eval_rmse, plot_reconstruction
    pred = DGraphSurfacePredictor(graph, underlyings)
    rmse = eval_rmse(pred, qry_feats, asset_ids, targets, ctx_max, [5,20,50])
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def eval_rmse(
    predictor,
    qry_feats:  np.ndarray,              # (N, Q, Q_dim)
    asset_ids:  np.ndarray,              # (N, Q)
    targets:    np.ndarray,              # (N, Q)
    ctx_max:    int,
    ctx_sizes:  list[int],
    n_reps:     int = 8,
    valid_mask: np.ndarray | None = None, # (N, tgt_pool) bool; None = all valid
) -> dict[int, float]:
    """
    RMSE (IV units) on the target pool (positions ctx_max:) for each context size.

    For parametric predictors (dgraph) the context is ignored; the RMSE will
    be constant across ctx_sizes, serving as a flat baseline on the same plot.
    """
    rng     = np.random.default_rng(0)
    tgt_true = targets[:, ctx_max:]
    if valid_mask is None:
        valid_mask = np.ones(tgt_true.shape, dtype=bool)

    results = {}
    for nc in ctx_sizes:
        nc = min(nc, ctx_max)
        total_sq = total_n = 0
        for _ in range(n_reps):
            perm     = rng.permutation(ctx_max)[:nc]
            pred     = predictor.predict(
                qry_feats[:, perm], asset_ids[:, perm], targets[:, perm],
                qry_feats, asset_ids,
            )
            err2 = (pred[:, ctx_max:] - tgt_true) ** 2
            total_sq += float((err2 * valid_mask).sum())
            total_n  += int(valid_mask.sum())
        results[nc] = float(np.sqrt(total_sq / max(total_n, 1)))
    return results


def plot_reconstruction(
    predictor,
    qry_feats:    np.ndarray,         # (N, Q, Q_dim)
    asset_ids:    np.ndarray,         # (N, Q)
    targets:      np.ndarray,         # (N, Q)
    ctx_max:      int,
    ctx_sizes,
    day_idx:      int = 0,
    asset_names:  list[str] | None = None,
    feat_names:   list[str] | None = None,
    feat_dim_x:   int = 0,
    feat_dim_grp: int = 1,
    out_path:     str | None = None,
) -> plt.Figure:
    """
    Reconstruction plot for one day: dashed = true IV, solid = predicted IV.
    Colours encode the grouping feature (default dim 1 = maturity T).
    """
    n_assets = int(asset_ids.max()) + 1
    names  = asset_names or [f"asset_{i}" for i in range(n_assets)]
    fnames = feat_names  or ["x", "group"]

    qf   = qry_feats[[day_idx]]
    qa   = asset_ids[[day_idx]]
    true = targets[day_idx]

    group_vals = np.unique(qf[0, ctx_max:, feat_dim_grp])
    group_vals = group_vals[group_vals > 0]
    col_map    = {v: plt.cm.viridis(i / max(len(group_vals) - 1, 1))
                  for i, v in enumerate(group_vals)}
    tgt_mask   = np.arange(qry_feats.shape[1]) >= ctx_max

    fig, axes = plt.subplots(n_assets, len(ctx_sizes),
                              figsize=(4.5 * len(ctx_sizes), 3 * n_assets),
                              sharey="row")
    if n_assets == 1:
        axes = axes[np.newaxis, :]

    for ci, nc in enumerate(ctx_sizes):
        perm = np.random.default_rng(ci).permutation(ctx_max)[:nc]
        pred = predictor.predict(
            qf[:, perm], qa[:, perm], targets[[day_idx]][:, perm], qf, qa,
        )[0]
        for a in range(n_assets):
            ax = axes[a, ci]
            for gv in group_vals:
                m = tgt_mask & (qa[0] == a) & (qf[0, :, feat_dim_grp] == gv)
                if m.sum() < 2:
                    continue
                order = np.argsort(qf[0, m, feat_dim_x])
                x = qf[0, m, feat_dim_x][order]
                ax.plot(x, true[m][order], "--", color=col_map[gv], lw=1.2, alpha=0.6)
                ax.plot(x, pred[m][order],  "-", color=col_map[gv], lw=1.5)
            if a == 0:
                ax.set_title(f"n_ctx={nc}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(names[a], fontsize=8)
            ax.set_xlabel(fnames[feat_dim_x], fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Reconstruction  (dashed=true, solid=pred)", fontsize=11)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.tight_layout()
    return fig


def _maybe_save(fig: plt.Figure, path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.tight_layout()
