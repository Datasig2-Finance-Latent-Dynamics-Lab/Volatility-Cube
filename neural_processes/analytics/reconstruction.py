from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from neural_processes.data.base import SurfaceDataset
from utils.pricing import bs_call_from_iv


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
    """TODO.

    Args:
        model: TODO.
        dataset: TODO.
        day_idx: TODO.
        ctx_sizes: TODO.
        feat_dim_x: TODO.
        feat_dim_group: TODO.
        out_path: TODO.

    Returns:
        TODO.
    """
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
    """TODO.

    Args:
        val_rmse: TODO.
        ood_rmse: TODO.
        out_path: TODO.

    Returns:
        TODO.
    """
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
    """TODO.

    Args:
        rmse_by_val: TODO.
        feat_name: TODO.
        n_ctx: TODO.
        out_path: TODO.

    Returns:
        TODO.
    """
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
    """TODO.

    Args:
        results: TODO.
        asset_names: TODO.
        n_ctx: TODO.
        out_path: TODO.

    Returns:
        TODO.
    """
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


def plot_call_reconstruction(model, dataset, day_idx, ctx_sizes=(5, 20, 50, 100),
                             feat_dim_lm=0, feat_dim_T=1, out_path=None):
    """
    Like plot_reconstruction but in forward-normalised call-price space.

    Call prices are computed as bs_call_from_iv(iv, lm, T, forward=1, discount=1),
    giving C / (F * discount) — a purely (iv, lm, T)-driven quantity that strips
    out the absolute forward level and interest-rate effects.

    Dashed lines = true quotes (converted from IV), solid lines = model predictions.
    Colours encode maturity T (same as plot_reconstruction).
    """
    ctx_max     = dataset.ctx_max
    n_assets    = dataset.n_assets
    asset_names = dataset.meta.get("asset_names", [f"asset_{i}" for i in range(n_assets)])
    feat_names  = dataset.meta.get("query_feat_names", ["lm", "T"])

    qf      = dataset.query_feats[[day_idx]]   # (1, N, Q)
    qa      = dataset.asset_ids[[day_idx]]     # (1, N)
    true_iv = dataset.targets[day_idx]         # (N,)

    lm_all = qf[0, :, feat_dim_lm]
    T_all  = qf[0, :, feat_dim_T]
    true_call = bs_call_from_iv(true_iv, lm_all, T_all)

    has_quotes = dataset.bid is not None and dataset.ask is not None
    bid_all = dataset.bid[day_idx]  if has_quotes else None
    ask_all = dataset.ask[day_idx]  if has_quotes else None

    group_vals = np.unique(qf[0, ctx_max:, feat_dim_T])
    group_vals = group_vals[group_vals > 0]
    cmap    = plt.cm.viridis
    col_map = {v: cmap(i / max(len(group_vals) - 1, 1)) for i, v in enumerate(group_vals)}
    tgt_mask = np.arange(dataset.n_points) >= ctx_max

    fig, axes = plt.subplots(n_assets, len(ctx_sizes),
                             figsize=(4.5 * len(ctx_sizes), 3 * n_assets), sharey="row")
    if n_assets == 1:
        axes = axes[np.newaxis, :]

    for ci, nc in enumerate(ctx_sizes):
        perm    = np.random.default_rng(ci).permutation(ctx_max)[:nc]
        pred_iv = model.predict(
            qf[:, perm], qa[:, perm],
            dataset.targets[[day_idx]][:, perm],
            qf, qa,
        )[0]
        pred_call = bs_call_from_iv(pred_iv, lm_all, T_all)

        for a in range(n_assets):
            ax = axes[a, ci]
            for gv in group_vals:
                m = tgt_mask & (qa[0] == a) & (qf[0, :, feat_dim_T] == gv)
                if m.sum() < 2:
                    continue
                order = np.argsort(lm_all[m])
                x = lm_all[m][order]
                col = col_map[gv]

                if has_quotes:
                    bid_v = bid_all[m][order]
                    ask_v = ask_all[m][order]
                    valid = np.isfinite(bid_v) & np.isfinite(ask_v)
                    if valid.any():
                        ax.fill_between(x[valid], bid_v[valid], ask_v[valid],
                                        color=col, alpha=0.18, linewidth=0)
                        ax.scatter(x[valid], bid_v[valid], s=8, color=col,
                                   marker="v", zorder=3, alpha=0.7)
                        ax.scatter(x[valid], ask_v[valid], s=8, color=col,
                                   marker="^", zorder=3, alpha=0.7)

                ax.plot(x, true_call[m][order],  "--", color=col, lw=1.2, alpha=0.6)
                ax.plot(x, pred_call[m][order],   "-", color=col, lw=1.5)

            if a == 0:
                ax.set_title(f"n_ctx={nc}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(asset_names[a], fontsize=8)
            ax.set_xlabel(feat_names[feat_dim_lm], fontsize=7)
            ax.tick_params(labelsize=6)

    bid_ask_note = "  ▼bid  ▲ask (shaded=spread)" if has_quotes else ""
    fig.suptitle(
        f"Call price reconstruction — forward-normalised C/F  (dashed=IV-quotes, solid=pred{bid_ask_note})",
        fontsize=10,
    )
    _maybe_save(fig, out_path)
    return fig


def compute_atm_and_skew(model, dataset, indices, target_mat=0.5, dk=0.05, n_ctx=None):
    """
    Predict ATM implied vol and skew from the model for each day in `indices`.

    Uses the smooth predicted surface (not raw data), so results are clean even
    on sparse observation days.

    Returns
    -------
    atm_iv : (N, n_assets)  σ at k=0, T=target_mat
    skew   : (N, n_assets)  ∂σ/∂k|_{k=0} ≈ (σ(+dk) − σ(−dk)) / (2·dk)
    """
    n_ctx    = min(n_ctx or dataset.ctx_max, dataset.ctx_max)
    n_days   = len(indices)
    n_assets = dataset.n_assets

    z = model.encode_dataset(dataset, indices, n_ctx)  # (N, n_assets, d_latent)

    # Build (3·n_assets) query points: (-dk, 0, +dk) × target_mat × each asset
    qry_pts, aid_pts = [], []
    for a in range(n_assets):
        for lm in (-dk, 0.0, dk):
            qry_pts.append([lm, float(target_mat)])
            aid_pts.append(a)
    qry_pts = np.array(qry_pts, dtype=np.float32)   # (3·n_assets, 2)
    aid_pts = np.array(aid_pts, dtype=np.int64)      # (3·n_assets,)

    qry_batch  = np.tile(qry_pts[None],  (n_days, 1, 1))  # (N, 3·n_assets, 2)
    aids_batch = np.tile(aid_pts[None],  (n_days, 1))      # (N, 3·n_assets)

    preds = model.decode_latent(z, qry_batch, aids_batch)  # (N, 3·n_assets)
    preds = preds.reshape(n_days, n_assets, 3)             # (N, n_assets, {-dk,0,+dk})

    atm_iv = preds[:, :, 1]
    skew   = (preds[:, :, 2] - preds[:, :, 0]) / (2.0 * dk)
    return atm_iv, skew


def plot_ssr_evolution(
    model,
    dataset,
    val_idx,
    *,
    log_fwd=None,
    target_mat=0.5,
    dk=0.05,
    n_ctx=None,
    asset_names=None,
    smooth_window=10,
    out_path=None,
):
    """
    Plot the evolution of ATM vol, skew, and (optionally) Skew Stickiness Ratio
    over the validation period.

    SSR is defined as  Δσ_ATM / (skew × Δlog F).  It equals 0 in the
    sticky-strike regime and 1 in the sticky-delta regime.  It is only
    computed when `log_fwd` is provided.

    Parameters
    ----------
    model       : FittedCNP
    dataset     : SurfaceDataset  (val split used)
    val_idx     : 1-D int array — indices into dataset for the validation days
    log_fwd     : (N_val, n_assets) log forward prices for each val day, or None.
                  Pass ``dataset.meta["log_fwd"][val_idx]`` for GroupTech data.
    target_mat  : maturity at which ATM vol and skew are evaluated (years)
    dk          : finite-difference step for skew (log-moneyness units)
    n_ctx       : context size used when encoding (default: dataset.ctx_max)
    asset_names : list[str] | None
    smooth_window : rolling-mean window for the SSR line (set to 1 to disable)
    out_path    : file path to save, or None
    """
    import pandas as pd

    n_assets = dataset.n_assets
    names    = asset_names or dataset.meta.get("asset_names",
                                               [f"asset_{i}" for i in range(n_assets)])
    colors   = plt.cm.tab10(np.linspace(0, 0.9, min(n_assets, 10)))

    # x-axis: use date labels when available
    raw_dates = dataset.meta.get("dates")
    if raw_dates is not None:
        x_labels = [raw_dates[i] for i in val_idx]
    else:
        x_labels = None
    x = np.arange(len(val_idx))

    atm_iv, skew = compute_atm_and_skew(
        model, dataset, val_idx, target_mat=target_mat, dk=dk, n_ctx=n_ctx
    )

    has_ssr  = log_fwd is not None
    n_panels = 3 if has_ssr else 2
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(13, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    def _plot_lines(ax, data, ylabel, hlines=()):
        for a in range(n_assets):
            ax.plot(x, data[:, a], color=colors[a % 10], lw=1.5, label=names[a])
        for yval, ls, label in hlines:
            ax.axhline(yval, color="k", lw=0.8, ls=ls, alpha=0.5, label=label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)

    # ── Panel 1: ATM vol ──────────────────────────────────────────────────────
    _plot_lines(axes[0], atm_iv, f"ATM vol  (T={target_mat})")
    axes[0].set_title(
        f"Vol surface dynamics — validation period  (T={target_mat} yr)",
        fontsize=11,
    )
    axes[0].legend(fontsize=7, ncol=min(n_assets, 4), loc="upper right")

    # ── Panel 2: Skew ─────────────────────────────────────────────────────────
    _plot_lines(axes[1], skew, "Skew  ∂σ/∂k|_{k=0}",
                hlines=[(0.0, "--", None)])

    # ── Panel 3: SSR (only when log_fwd provided) ────────────────────────────
    if has_ssr:
        d_log_f = np.diff(log_fwd,  axis=0)   # (N_val-1, n_assets)
        d_atm   = np.diff(atm_iv,   axis=0)   # (N_val-1, n_assets)
        denom   = skew[:-1] * d_log_f
        ssr_raw = np.where(np.abs(denom) > 1e-5, d_atm / denom, np.nan)

        ax = axes[2]
        x_mid = x[:-1] + 0.5
        for a in range(n_assets):
            s = ssr_raw[:, a]
            ax.plot(x_mid, s, color=colors[a % 10], lw=0.7, alpha=0.35)
            s_smooth = (
                pd.Series(s)
                .rolling(smooth_window, center=True, min_periods=1)
                .mean()
                .values
            )
            ax.plot(x_mid, s_smooth, color=colors[a % 10], lw=2.0,
                    label=names[a])

        ax.axhline(0.0, color="k",    lw=1.0, ls="--", alpha=0.6,
                   label="sticky-strike (SSR=0)")
        ax.axhline(1.0, color="grey", lw=1.0, ls="--", alpha=0.6,
                   label="sticky-delta (SSR=1)")
        ax.set_ylabel("SSR", fontsize=9)
        ax.set_ylim(-3, 4)
        ax.legend(fontsize=7, ncol=min(n_assets + 2, 5), loc="upper right")
        ax.grid(True, alpha=0.25)

    # ── x-axis ticks ─────────────────────────────────────────────────────────
    ax_bot = axes[-1]
    ax_bot.set_xlabel("Validation day", fontsize=9)
    if x_labels is not None:
        step = max(1, len(x_labels) // 10)
        ticks = x[::step]
        ax_bot.set_xticks(ticks)
        ax_bot.set_xticklabels(
            [x_labels[i] for i in ticks], rotation=30, ha="right", fontsize=7
        )

    fig.tight_layout()
    _maybe_save(fig, out_path)
    return fig


def _maybe_save(fig, path):
    if path:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.tight_layout()
