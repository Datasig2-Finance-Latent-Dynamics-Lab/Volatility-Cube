"""
Shared helpers used by all NP example run scripts.

Each run.py handles its own data loading and training (those differ per
experiment), then delegates the standard analytics block to
run_standard_analytics() and training curve plotting to plot_training_curve().
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from neural_processes.analytics import (
    eval_rmse_vs_ctx, eval_per_feature_rmse, eval_zeroshot,
    plot_reconstruction, plot_call_reconstruction,
    plot_rmse_vs_ctx, plot_per_feature_rmse, plot_zeroshot,
)


def plot_training_curve(history: dict, out: Path, title: str = "Training curve") -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["train_rmse"], lw=1, alpha=0.8, label="Train RMSE (norm)")
    if history["val_rmse"]:
        vx, vy = zip(*history["val_rmse"])
        ax.plot(vx, vy, "o-", color="C1", ms=4, label="Val RMSE (norm)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE"); ax.set_yscale("log")
    ax.legend(); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_standard_analytics(
    model,
    dataset,
    ac,
    out: Path,
    *,
    ood_ds=None,
    results_csv: str | None = None,
    experiment: str | None = None,
) -> None:
    """
    Standard analytics suite for a trained NP model.

    Runs: reconstruction, call-price reconstruction, RMSE vs context size,
    per-maturity RMSE, and zero-shot asset reconstruction.  Saves all plots
    and a JSON summary to `out`.

    Parameters
    ----------
    model   : FittedCNP
    dataset : SurfaceDataset  (val split is used for evaluation)
    ac      : AnalyticsConfig (needs ctx_sizes_recon, ctx_sizes_rmse_curve,
                               n_ctx_per_maturity, n_ctx_zeroshot)
    out     : output directory (Path)
    ood_ds  : optional OOD SurfaceDataset; if provided, ood RMSE is computed
              and shown alongside val RMSE
    """
    val_idx     = dataset.val_idx()
    n_assets    = dataset.n_assets
    asset_names = dataset.meta.get("asset_names", [f"asset_{i}" for i in range(n_assets)])
    feat_names  = dataset.meta.get("query_feat_names", ["x", "group"])

    # Reconstruction plots use the last eval day for a representative picture.
    last_val_day = val_idx[-1]

    print("Reconstruction plot...")
    plot_reconstruction(model, dataset, last_val_day, ctx_sizes=ac.ctx_sizes_recon,
                        out_path=str(out / "reconstruction.png"))
    plt.close("all")

    print("Call price reconstruction plot...")
    plot_call_reconstruction(model, dataset, last_val_day, ctx_sizes=ac.ctx_sizes_recon,
                             out_path=str(out / "call_reconstruction.png"))
    plt.close("all")

    print("RMSE vs context size...")
    val_rmse = eval_rmse_vs_ctx(model, dataset, val_idx, ac.ctx_sizes_rmse_curve)
    ood_rmse = None
    if ood_ds is not None:
        ood_idx  = np.arange(ood_ds.n_days)
        ood_rmse = eval_rmse_vs_ctx(model, ood_ds, ood_idx, ac.ctx_sizes_rmse_curve)
        last_nc  = ac.ctx_sizes_rmse_curve[-1]
        print(f"  OOD/Val ratio at n_ctx={last_nc}: "
              f"{ood_rmse[last_nc] / max(val_rmse[last_nc], 1e-8):.2f}x")
    plot_rmse_vs_ctx(val_rmse, ood_rmse, out_path=str(out / "rmse_vs_ctx.png"))
    plt.close("all")
    eval_metrics: dict = {
        "n_eval_days": len(val_idx),
        "rmse_vs_ctx": {"val": val_rmse, "ood": ood_rmse},
    }
    with open(out / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    if results_csv is not None and experiment is not None:
        from utils.results_csv import upsert_results_csv
        row: dict = {
            "experiment":  experiment,
            "model":       "cnp",
            "n_eval_days": len(val_idx),
        }
        for ctx, rmse in val_rmse.items():
            row[f"rmse_ctx_{ctx}"] = round(float(rmse), 8)
        # Largest context size as the headline "test_rmse" for cross-experiment comparison.
        largest_ctx = max(val_rmse.keys())
        row["test_rmse"] = round(float(val_rmse[largest_ctx]), 8)
        upsert_results_csv(results_csv, [row])

    print("Per-maturity RMSE...")
    mat_rmse = eval_per_feature_rmse(model, dataset, val_idx,
                                      n_ctx=ac.n_ctx_per_maturity, feat_dim=1)
    plot_per_feature_rmse(mat_rmse, feat_name=feat_names[1] if len(feat_names) > 1 else "maturity",
                          n_ctx=ac.n_ctx_per_maturity,
                          out_path=str(out / "per_maturity_rmse.png"))
    plt.close("all")

    print("Zero-shot test...")
    zs = eval_zeroshot(model, dataset, val_idx[:50], n_ctx=ac.n_ctx_zeroshot)
    plot_zeroshot(zs, asset_names=asset_names, n_ctx=ac.n_ctx_zeroshot,
                  out_path=str(out / "zeroshot.png"))
    plt.close("all")
    for a, nm in enumerate(asset_names):
        print(f"  {nm}: baseline={zs['baseline'][a]:.4f}  zs={zs['zeroshot'][a]:.4f}")
