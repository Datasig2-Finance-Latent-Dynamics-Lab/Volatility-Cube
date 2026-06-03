"""
Analytics for the increment (delta) CNP experiment — hourly data.

Run from repo root:
    .venv/bin/python3 -m neural_processes.examples.vol_surface_increment_hourly.analytics
    .venv/bin/python3 -m neural_processes.examples.vol_surface_increment_hourly.analytics --out_dir results/vol_surface_increment_hourly
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import Config
from neural_processes.data import load_grouptech, compute_bspline_prior
from neural_processes.models.cnp import FittedDeltaCNP
from .._shared import plot_training_curve


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_rmse_vs_ctx(model: FittedDeltaCNP, dataset, indices, ctx_sizes, n_reps=8):
    """
    RMSE on absolute IV (target pool) for each context size.
    Also computes prior-only baseline RMSE (model receives zero context).
    """
    ctx_max   = dataset.ctx_max
    rng       = np.random.default_rng(0)
    valid_tgt = dataset.query_feats[indices, ctx_max:, 1] > 0  # (N, tgt_pool)

    prior_tgt = dataset.prior_targets[indices, ctx_max:]        # (N, tgt_pool)
    true_tgt  = dataset.targets[indices, ctx_max:]              # (N, tgt_pool)

    # Prior baseline: RMSE of just using prior as prediction
    prior_err2   = (prior_tgt - true_tgt) ** 2
    prior_valid  = valid_tgt & np.isfinite(prior_tgt) & np.isfinite(true_tgt)
    prior_rmse   = float(np.sqrt(prior_err2[prior_valid].sum()
                                 / max(prior_valid.sum(), 1)))

    model_rmse = {}
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
                obs_prior_iv=dataset.prior_targets[indices][:, perm],
                qry_prior_iv=dataset.prior_targets[indices],
            )
            v    = valid_tgt & np.isfinite(pred[:, ctx_max:])
            err2 = np.where(v, (pred[:, ctx_max:] - true_tgt) ** 2, 0.0)
            total_sq += float(err2.sum())
            total_n  += int(v.sum())
        model_rmse[nc] = float(np.sqrt(total_sq / max(total_n, 1)))

    return {"model": model_rmse, "prior_baseline": prior_rmse}


def eval_per_maturity_rmse(model: FittedDeltaCNP, dataset, indices,
                           n_ctx=100, n_reps=6):
    """RMSE on absolute IV broken down by maturity, plus per-maturity prior baseline."""
    ctx_max     = dataset.ctx_max
    rng         = np.random.default_rng(0)
    tgt_T       = dataset.query_feats[indices[0], ctx_max:, 1]
    unique_T    = np.unique(tgt_T)
    unique_T    = unique_T[unique_T > 0]

    true_tgt  = dataset.targets[indices, ctx_max:]
    prior_tgt = dataset.prior_targets[indices, ctx_max:]
    feat_T    = dataset.query_feats[indices, ctx_max:, 1]

    model_by_T  = {v: [] for v in unique_T}
    prior_by_T  = {v: [] for v in unique_T}

    for _ in range(n_reps):
        perm = rng.permutation(ctx_max)[:n_ctx]
        pred = model.predict(
            dataset.query_feats[indices][:, perm],
            dataset.asset_ids[indices][:, perm],
            dataset.targets[indices][:, perm],
            dataset.query_feats[indices],
            dataset.asset_ids[indices],
            obs_prior_iv=dataset.prior_targets[indices][:, perm],
            qry_prior_iv=dataset.prior_targets[indices],
        )
        pred_t = pred[:, ctx_max:]
        for tv in unique_T:
            mask = feat_T == tv
            vm = mask & np.isfinite(pred_t)
            if vm.any():
                model_by_T[tv].append(float(((pred_t - true_tgt) ** 2)[vm].mean()))
            pm = mask & np.isfinite(prior_tgt)
            if pm.any():
                prior_by_T[tv].append(float(((prior_tgt - true_tgt) ** 2)[pm].mean()))

    model_rmse = {float(tv): float(np.sqrt(np.mean(model_by_T[tv])))
                  for tv in unique_T if model_by_T[tv]}
    prior_rmse = {float(tv): float(np.sqrt(np.mean(prior_by_T[tv])))
                  for tv in unique_T if prior_by_T[tv]}
    return model_rmse, prior_rmse


def eval_zeroshot_delta(model: FittedDeltaCNP, dataset, indices, n_ctx=50, n_reps=10):
    """
    Per-asset RMSE on absolute IV for three conditions:
      - prior_only : use prior IV directly, no model inference
      - baseline   : n_ctx observations from all assets
      - zeroshot   : n_ctx observations excluding the target asset
    Returns {'prior': array(A,), 'baseline': array(A,), 'zeroshot': array(A,)}.
    """
    ctx_max  = dataset.ctx_max
    n_assets = dataset.n_assets
    rng      = np.random.default_rng(0)

    aid_tgt   = dataset.asset_ids[indices, ctx_max:]
    valid_tgt = dataset.query_feats[indices, ctx_max:, 1] > 0
    true_tgt  = dataset.targets[indices, ctx_max:]
    prior_tgt = dataset.prior_targets[indices, ctx_max:]

    prior_arr    = np.zeros(n_assets)
    baseline_arr = np.zeros(n_assets)
    zeroshot_arr = np.zeros(n_assets)

    qry_f      = dataset.query_feats[indices]
    qry_a      = dataset.asset_ids[indices]
    pool_aids  = dataset.asset_ids[indices, :ctx_max]

    for excl in range(n_assets):
        mask = (aid_tgt == excl) & valid_tgt & np.isfinite(prior_tgt) & np.isfinite(true_tgt)
        if not mask.any():
            continue

        prior_err2 = (prior_tgt - true_tgt) ** 2
        prior_arr[excl] = float(np.sqrt(
            prior_err2[mask].sum() / max(int(mask.sum()), 1)
        ))

        sq_bl, sq_zs = [], []
        for _ in range(n_reps):
            perm    = rng.permutation(ctx_max)[:n_ctx]
            pred_bl = model.predict(
                qry_f[:, perm], qry_a[:, perm],
                dataset.targets[indices][:, perm],
                qry_f, qry_a,
                obs_prior_iv=dataset.prior_targets[indices][:, perm],
                qry_prior_iv=dataset.prior_targets[indices],
            )[:, ctx_max:]
            sq_bl.append(float(((pred_bl - true_tgt) ** 2)[mask].mean()))

            obs_f_list, obs_a_list, obs_t_list, obs_p_list = [], [], [], []
            for b in range(len(indices)):
                avail  = np.where(pool_aids[b] != excl)[0]
                if len(avail) == 0:
                    avail = np.arange(ctx_max)
                chosen = avail[rng.permutation(len(avail))[:n_ctx]]
                obs_f_list.append(dataset.query_feats[indices[b], chosen])
                obs_a_list.append(dataset.asset_ids[indices[b], chosen])
                obs_t_list.append(dataset.targets[indices[b], chosen])
                obs_p_list.append(dataset.prior_targets[indices[b], chosen])
            obs_f = np.stack(obs_f_list)
            obs_a = np.stack(obs_a_list)
            obs_t = np.stack(obs_t_list)
            obs_p = np.stack(obs_p_list)

            pred_zs = model.predict(
                obs_f, obs_a, obs_t, qry_f, qry_a,
                obs_prior_iv=obs_p,
                qry_prior_iv=dataset.prior_targets[indices],
            )[:, ctx_max:]
            sq_zs.append(float(((pred_zs - true_tgt) ** 2)[mask].mean()))

        baseline_arr[excl] = float(np.sqrt(np.mean(sq_bl)))
        zeroshot_arr[excl]  = float(np.sqrt(np.mean(sq_zs)))

    return {"prior": prior_arr, "baseline": baseline_arr, "zeroshot": zeroshot_arr}


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_rmse_vs_ctx(results, out_path=None):
    model_rmse = results["model"]
    prior_rmse = results["prior_baseline"]
    ctx = sorted(model_rmse)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ctx, [model_rmse[c] for c in ctx], "o-", lw=2, color="C0", label="Delta-CNP")
    ax.axhline(prior_rmse, color="C1", lw=1.5, ls="--",
               label=f"Prior-only baseline ({prior_rmse:.4f})")
    ax.set_xlabel("Context size"); ax.set_ylabel("RMSE (absolute IV)")
    if ctx:
        ax.set_xlim(left=min(ctx) * 0.8, right=max(ctx) * 1.25)
    ax.set_xscale("log"); ax.legend()
    ax.set_title("RMSE vs context size — increment model (hourly)")
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_maturity_rmse(model_rmse, prior_rmse, n_ctx=100, out_path=None):
    vals = sorted(model_rmse)
    x    = np.arange(len(vals))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, [model_rmse[v] for v in vals], w, color="C0", alpha=0.85,
           label=f"Delta-CNP (n_ctx={n_ctx})")
    ax.bar(x + w/2, [prior_rmse.get(v, 0) for v in vals], w, color="C1", alpha=0.85,
           label="Prior-only baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in vals], fontsize=9)
    ax.set_xlabel("Maturity T (years)"); ax.set_ylabel("RMSE (absolute IV)")
    ax.set_title(f"Per-maturity RMSE — increment model hourly  (n_ctx={n_ctx})")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_zeroshot_increment(results, asset_names=None, n_ctx=50, out_path=None):
    """Grouped bar chart: prior only, full CNP, zero-shot CNP per asset."""
    n_assets = len(results["prior"])
    names    = asset_names or [f"asset_{i}" for i in range(n_assets)]
    x        = np.arange(n_assets)
    w        = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, results["prior"],    w, label="Prior only",               color="C1", alpha=0.85)
    ax.bar(x,     results["baseline"], w, label=f"Full CNP (n_ctx={n_ctx})", color="C0", alpha=0.85)
    ax.bar(x + w, results["zeroshot"], w, label="Zero-shot CNP (excl. asset)", color="C3", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Excl:\n{n}" for n in names], fontsize=8)
    ax.set_ylabel("RMSE on excluded asset (absolute IV)")
    ax.set_title("Zero-Shot Asset Reconstruction — increment model (hourly)")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction(model: FittedDeltaCNP, dataset, day_idx,
                        ctx_sizes=(5, 20, 50, 100), out_path=None):
    """Reconstruction plot for a single observation."""
    ctx_max     = dataset.ctx_max
    n_assets    = dataset.n_assets
    asset_names = dataset.meta.get("asset_names", [f"asset_{i}" for i in range(n_assets)])

    qf       = dataset.query_feats[[day_idx]]
    qa       = dataset.asset_ids[[day_idx]]
    true_iv  = dataset.targets[day_idx]
    prior_iv = dataset.prior_targets[day_idx]

    group_vals = np.unique(qf[0, ctx_max:, 1])
    group_vals = group_vals[group_vals > 0]
    cmap    = plt.cm.viridis
    col_map = {v: cmap(i / max(len(group_vals) - 1, 1)) for i, v in enumerate(group_vals)}
    tgt_mask = np.arange(dataset.n_points) >= ctx_max

    fig, axes = plt.subplots(n_assets, len(ctx_sizes),
                             figsize=(4.5 * len(ctx_sizes), 3 * n_assets), sharey="row")
    if n_assets == 1:
        axes = axes[np.newaxis, :]

    for ci, nc in enumerate(ctx_sizes):
        perm = np.random.default_rng(ci).permutation(ctx_max)[:nc]
        pred_iv = model.predict(
            qf[:, perm], qa[:, perm],
            dataset.targets[[day_idx]][:, perm],
            qf, qa,
            obs_prior_iv=dataset.prior_targets[[day_idx]][:, perm],
            qry_prior_iv=dataset.prior_targets[[day_idx]],
        )[0]

        for a in range(n_assets):
            ax = axes[a, ci]
            for gv in group_vals:
                m = tgt_mask & (qa[0] == a) & (qf[0, :, 1] == gv)
                if m.sum() < 2:
                    continue
                order = np.argsort(qf[0, m, 0])
                x = qf[0, m, 0][order]
                ax.plot(x, true_iv[m][order],  "--", color=col_map[gv], lw=1.2, alpha=0.6)
                ax.plot(x, prior_iv[m][order], ":", color=col_map[gv], lw=1.2, alpha=0.5)
                ax.plot(x, pred_iv[m][order],   "-", color=col_map[gv], lw=1.5)
            if a == 0:
                ax.set_title(f"n_ctx={nc}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(asset_names[a], fontsize=8)
            ax.set_xlabel("log-moneyness", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Reconstruction (dashed=true, dotted=prior, solid=model)", fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: Config, model=None, dataset=None):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    if dataset is None:
        print("Loading data (hourly)...")
        dataset = load_grouptech(
            cfg.data.csv_path,
            obs_col=cfg.data.obs_col,
            n_eval_days=cfg.data.n_eval_days,
            seed=cfg.seed,
        )
        print("Computing BSpline priors...")
        dataset.prior_targets = compute_bspline_prior(dataset)

    if model is None:
        model_path = out / "model.pt"
        print(f"Loading model from {model_path}...")
        model = FittedDeltaCNP.load(str(model_path))

    val_idx     = dataset.val_idx()
    asset_names = dataset.meta.get("asset_names",
                                   [f"asset_{i}" for i in range(dataset.n_assets)])

    print("RMSE vs context size...")
    rmse_results = eval_rmse_vs_ctx(
        model, dataset, val_idx, cfg.analytics.ctx_sizes_rmse_curve
    )
    plot_rmse_vs_ctx(rmse_results, out_path=str(out / "rmse_vs_ctx.png"))
    print(f"  Prior baseline RMSE: {rmse_results['prior_baseline']:.4f}")
    for nc, rmse in sorted(rmse_results["model"].items()):
        print(f"  n_ctx={nc:4d}: {rmse:.4f}")

    print("Per-maturity RMSE...")
    model_mat, prior_mat = eval_per_maturity_rmse(
        model, dataset, val_idx, n_ctx=cfg.analytics.n_ctx_per_maturity
    )
    plot_per_maturity_rmse(model_mat, prior_mat,
                           n_ctx=cfg.analytics.n_ctx_per_maturity,
                           out_path=str(out / "per_maturity_rmse.png"))

    print("Zero-shot evaluation...")
    zs_results = eval_zeroshot_delta(
        model, dataset, val_idx, n_ctx=cfg.analytics.n_ctx_zeroshot
    )
    plot_zeroshot_increment(
        zs_results, asset_names=asset_names,
        n_ctx=cfg.analytics.n_ctx_zeroshot,
        out_path=str(out / "zeroshot.png"),
    )
    for i, name in enumerate(asset_names):
        print(f"  {name}: prior={zs_results['prior'][i]:.4f}  "
              f"baseline={zs_results['baseline'][i]:.4f}  "
              f"zeroshot={zs_results['zeroshot'][i]:.4f}")

    print("Reconstruction plot...")
    last_val_obs = val_idx[-1]
    plot_reconstruction(model, dataset, last_val_obs,
                        ctx_sizes=cfg.analytics.ctx_sizes_recon,
                        out_path=str(out / "reconstruction.png"))

    eval_metrics = {
        "n_eval_obs":       len(val_idx),
        "prior_baseline":   rmse_results["prior_baseline"],
        "rmse_vs_ctx":      rmse_results["model"],
        "per_maturity_rmse": model_mat,
        "zeroshot": {
            "prior":    zs_results["prior"].tolist(),
            "baseline": zs_results["baseline"].tolist(),
            "zeroshot": zs_results["zeroshot"].tolist(),
        },
    }
    with open(out / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"\nAll outputs saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.out_dir: cfg.out_dir = args.out_dir

    main(cfg)
