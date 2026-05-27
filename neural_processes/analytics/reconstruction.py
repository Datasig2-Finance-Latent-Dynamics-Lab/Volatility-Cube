from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from neural_processes.data.base import SurfaceDataset


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_rmse_vs_ctx(model, dataset, indices, ctx_sizes, n_reps=8):
    """
    RMSE (original IV units) on the target pool for each context size.
    Zero-padded positions (T == 0 in raw query_feats) are excluded from the mean.
    """
    ctx_max   = dataset.ctx_max
    rng       = np.random.default_rng(0)
    # valid mask for target pool: T > 0 (raw, dim 1 of query_feats)
    valid_tgt = dataset.query_feats[indices, ctx_max:, 1] > 0   # (N, tgt_pool)
    results   = {}
    for nc in ctx_sizes:
        nc = min(nc, ctx_max)
        total_sq, total_n = 0.0, 0
        for _ in range(n_reps):
            perm = rng.permutation(ctx_max)[:nc]
            pred = model.predict(
                dataset.query_feats[indices][:, perm],
                dataset.asset_ids[indices][:, perm],
                dataset.targets[indices][:, perm],
                dataset.query_feats[indices],
                dataset.asset_ids[indices],
            )
            true  = dataset.targets[indices]
            err2  = (pred[:, ctx_max:] - true[:, ctx_max:]) ** 2
            total_sq += float((err2 * valid_tgt).sum())
            total_n  += int(valid_tgt.sum())
        results[nc] = float(np.sqrt(total_sq / max(total_n, 1)))
    return results


def eval_per_feature_rmse(model, dataset, indices, n_ctx=100, feat_dim=1, n_reps=6):
    """
    RMSE broken down by unique values of one query feature (default dim 1 = maturity T).
    Zero-padded positions (T == 0) are excluded from both the bucket list and the mean.
    """
    ctx_max     = dataset.ctx_max
    rng         = np.random.default_rng(0)
    # Exclude T=0 bucket (zero-padding artefact)
    tgt_feat    = dataset.query_feats[indices[0], ctx_max:, feat_dim]
    unique_vals = np.unique(tgt_feat)
    unique_vals = unique_vals[unique_vals > 0]   # drop zero-padded T=0 bucket
    sq_by_val   = {v: [] for v in unique_vals}

    for _ in range(n_reps):
        perm   = rng.permutation(ctx_max)[:n_ctx]
        pred   = model.predict(
            dataset.query_feats[indices][:, perm],
            dataset.asset_ids[indices][:, perm],
            dataset.targets[indices][:, perm],
            dataset.query_feats[indices],
            dataset.asset_ids[indices],
        )
        true   = dataset.targets[indices]
        pred_t = pred[:, ctx_max:]
        true_t = true[:, ctx_max:]
        feat_t = dataset.query_feats[indices, ctx_max:, feat_dim]
        err2   = (pred_t - true_t) ** 2
        for v in unique_vals:
            mask = feat_t == v
            if mask.any():
                sq_by_val[v].append(err2[mask].mean())

    return {float(v): float(np.sqrt(np.mean(sq_by_val[v])))
            for v in unique_vals if sq_by_val[v]}


def eval_zeroshot(model, dataset, indices, n_ctx=50, n_reps=10):
    """
    Per-asset RMSE when context excludes that asset (zero-shot) vs baseline.
    Zero-padded positions are excluded via a T > 0 validity mask.
    Returns {'baseline': array(A,), 'zeroshot': array(A,)}.
    """
    ctx_max  = dataset.ctx_max
    n_assets = dataset.n_assets
    aid_tgt  = dataset.asset_ids[indices, ctx_max:]
    # valid mask: exclude padded positions (T=0) from target pool evaluation
    valid_tgt_pool = dataset.query_feats[indices, ctx_max:, 1] > 0   # (N, tgt_pool)
    rng      = np.random.default_rng(0)
    baseline  = np.zeros(n_assets)
    zeroshot  = np.zeros(n_assets)

    for excl in range(n_assets):
        pool_aids = dataset.asset_ids[indices, :ctx_max]
        qry_f = dataset.query_feats[indices]
        qry_a = dataset.asset_ids[indices]
        true  = dataset.targets[indices]
        mask  = (aid_tgt == excl) & valid_tgt_pool   # (N, tgt_pool)
        if not mask.any():
            continue

        sq_zs, sq_bl = [], []
        for _ in range(n_reps):
            # zero-shot context (exclude asset excl)
            obs_f_list, obs_a_list, obs_t_list = [], [], []
            for b in range(len(indices)):
                avail  = np.where(pool_aids[b] != excl)[0]
                if len(avail) == 0:
                    avail = np.arange(ctx_max)
                chosen = avail[rng.permutation(len(avail))[:n_ctx]]
                obs_f_list.append(dataset.query_feats[indices[b], chosen])
                obs_a_list.append(dataset.asset_ids[indices[b], chosen])
                obs_t_list.append(dataset.targets[indices[b], chosen])
            obs_f = np.stack(obs_f_list)
            obs_a = np.stack(obs_a_list)
            obs_t = np.stack(obs_t_list)

            pred_zs  = model.predict(obs_f, obs_a, obs_t, qry_f, qry_a)[:, ctx_max:]
            true_tgt = true[:, ctx_max:]
            sq_zs.append(((pred_zs - true_tgt) ** 2)[mask].mean())

            # baseline context
            perm_b  = rng.permutation(ctx_max)[:n_ctx]
            pred_bl = model.predict(
                dataset.query_feats[indices][:, perm_b],
                dataset.asset_ids[indices][:, perm_b],
                dataset.targets[indices][:, perm_b],
                qry_f, qry_a,
            )[:, ctx_max:]
            sq_bl.append(((pred_bl - true_tgt) ** 2)[mask].mean())

        zeroshot[excl] = float(np.sqrt(np.mean(sq_zs)))
        baseline[excl] = float(np.sqrt(np.mean(sq_bl)))

    return {"baseline": baseline, "zeroshot": zeroshot}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_reconstruction(model, dataset, day_idx, ctx_sizes=(5, 20, 50, 100),
                        feat_dim_x=0, feat_dim_group=1, out_path=None):
    ctx_max     = dataset.ctx_max
    n_assets    = dataset.n_assets
    asset_names = dataset.meta.get("asset_names", [f"asset_{i}" for i in range(n_assets)])
    feat_names  = dataset.meta.get("query_feat_names", ["x", "group"])
    qf = dataset.query_feats[[day_idx]]
    qa = dataset.asset_ids[[day_idx]]
    true = dataset.targets[day_idx]
    group_vals = np.unique(qf[0, ctx_max:, feat_dim_group])
    group_vals = group_vals[group_vals > 0]   # exclude T=0 padding bucket
    cmap = plt.cm.viridis
    col_map = {v: cmap(i / max(len(group_vals) - 1, 1)) for i, v in enumerate(group_vals)}
    tgt_mask = np.arange(dataset.n_points) >= ctx_max

    fig, axes = plt.subplots(n_assets, len(ctx_sizes),
                              figsize=(4.5 * len(ctx_sizes), 3 * n_assets), sharey="row")
    if n_assets == 1:
        axes = axes[np.newaxis, :]

    for ci, nc in enumerate(ctx_sizes):
        perm = np.random.default_rng(ci).permutation(ctx_max)[:nc]
        pred = model.predict(qf[:, perm], qa[:, perm],
                             dataset.targets[[day_idx]][:, perm], qf, qa)[0]
        for a in range(n_assets):
            ax = axes[a, ci]
            for gv in group_vals:
                m = tgt_mask & (qa[0] == a) & (qf[0, :, feat_dim_group] == gv)
                if m.sum() < 2:
                    continue
                order = np.argsort(qf[0, m, feat_dim_x])
                x = qf[0, m, feat_dim_x][order]
                ax.plot(x, true[m][order], "--", color=col_map[gv], lw=1.2, alpha=0.6)
                ax.plot(x, pred[m][order],  "-", color=col_map[gv], lw=1.5)
            if a == 0:
                ax.set_title(f"n_ctx={nc}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(asset_names[a], fontsize=8)
            ax.set_xlabel(feat_names[feat_dim_x], fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Target-pool reconstruction  (dashed=true, solid=pred)", fontsize=11)
    _maybe_save(fig, out_path)
    return fig


def plot_rmse_vs_ctx(val_rmse, ood_rmse=None, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ctx = sorted(val_rmse)
    ax.plot(ctx, [val_rmse[c] for c in ctx], "o-", label="Val", color="C0", lw=2)
    if ood_rmse:
        ax.plot(ctx, [ood_rmse[c] for c in ctx], "s--", label="OOD", color="C3", lw=2)
    ax.set_xlabel("Context size"); ax.set_ylabel("RMSE (IV units)")
    ax.set_xscale("log"); ax.legend(); ax.set_title("RMSE vs context size")
    _maybe_save(fig, out_path)
    return fig


def plot_per_feature_rmse(rmse_by_val, feat_name="maturity", n_ctx=100, out_path=None):
    vals = sorted(rmse_by_val)
    rmse = [rmse_by_val[v] for v in vals]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(vals)), rmse,
                  color=plt.cm.plasma(np.linspace(0.1, 0.9, len(vals))))
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in vals], fontsize=9)
    ax.set_xlabel(feat_name); ax.set_ylabel("RMSE (IV units)")
    ax.set_title(f"Per-{feat_name} RMSE  (n_ctx={n_ctx})")
    for bar, v in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    _maybe_save(fig, out_path)
    return fig


def plot_zeroshot(results, asset_names=None, n_ctx=50, out_path=None):
    n_assets = len(results["baseline"])
    names    = asset_names or [f"asset_{i}" for i in range(n_assets)]
    x        = np.arange(n_assets)
    fig, ax  = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, results["baseline"], 0.35, label=f"Baseline (n_ctx={n_ctx})",
           color="C0", alpha=0.85)
    ax.bar(x + 0.18, results["zeroshot"],  0.35, label="Zero-shot (exclude asset)",
           color="C3", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Excl:\n{n}" for n in names])
    ax.set_ylabel("RMSE on excluded asset (IV units)")
    ax.set_title("Zero-Shot Asset Reconstruction")
    ax.legend()
    _maybe_save(fig, out_path)
    return fig


def _maybe_save(fig, path):
    if path:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.tight_layout()
