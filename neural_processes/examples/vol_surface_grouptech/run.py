"""
CNP experiment on Group Tech US market data.

Run from repo root:
    python -m neural_processes.examples.vol_surface_grouptech.run
    python -m neural_processes.examples.vol_surface_grouptech.run --device cpu --n_train 200 --n_epochs 50
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import Config
from neural_processes.data import load_grouptech
from neural_processes.models import Trainer, FittedCNP
from neural_processes.analytics import (
    eval_rmse_vs_ctx, eval_per_feature_rmse, eval_zeroshot,
    plot_reconstruction, plot_rmse_vs_ctx, plot_per_feature_rmse, plot_zeroshot,
)


def main(cfg: Config):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    # 1. Data
    print("Loading data...")
    dataset = load_grouptech(
        cfg.data.csv_path,
        n_train_days=cfg.data.n_train_days,
        n_val_days=cfg.data.n_val_days,
        seed=cfg.seed,
        val_frac=cfg.data.val_frac,
    )
    print(f"  Train: {(dataset.split==0).sum()}  Val: {(dataset.split==1).sum()}"
          f"  Assets: {dataset.n_assets}  Points/day (max): {dataset.n_points}")

    # 2. Train
    print("\nTraining...")
    model, history = Trainer(cfg).train(dataset)
    model.save(str(out / "model.pt"))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["train_rmse"], lw=1, alpha=0.8, label="Train RMSE (norm)")
    if history["val_rmse"]:
        vx, vy = zip(*history["val_rmse"])
        ax.plot(vx, vy, "o-", color="C1", ms=4, label="Val RMSE (norm)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE"); ax.set_yscale("log")
    ax.legend(); ax.set_title("Training curve")
    fig.tight_layout(); fig.savefig(out / "training_curve.png", dpi=120); plt.close(fig)

    # 3. Analytics
    ac          = cfg.analytics
    val_idx     = dataset.val_idx()
    n_assets    = dataset.n_assets
    asset_names = dataset.meta.get("asset_names", [f"asset_{i}" for i in range(n_assets)])
    feat_names  = dataset.meta.get("query_feat_names", ["x", "group"])

    print("\nReconstruction plot...")
    plot_reconstruction(model, dataset, val_idx[0], ctx_sizes=ac.ctx_sizes_recon,
                        out_path=str(out / "reconstruction.png"))
    plt.close("all")

    print("RMSE vs context size...")
    val_rmse = eval_rmse_vs_ctx(model, dataset, val_idx, ac.ctx_sizes_rmse_curve)
    plot_rmse_vs_ctx(val_rmse, ood_rmse=None, out_path=str(out / "rmse_vs_ctx.png"))
    plt.close("all")
    with open(out / "rmse_vs_ctx.json", "w") as f:
        json.dump({"val": val_rmse}, f, indent=2)

    print("Per-maturity RMSE...")
    mat_rmse = eval_per_feature_rmse(model, dataset, val_idx,
                                      n_ctx=ac.n_ctx_per_maturity, feat_dim=1)
    plot_per_feature_rmse(mat_rmse, feat_name=feat_names[1] if len(feat_names) > 1 else "maturity",
                          n_ctx=ac.n_ctx_per_maturity, out_path=str(out / "per_maturity_rmse.png"))
    plt.close("all")

    print("Zero-shot test...")
    zs = eval_zeroshot(model, dataset, val_idx[:50], n_ctx=ac.n_ctx_zeroshot)
    plot_zeroshot(zs, asset_names=asset_names, n_ctx=ac.n_ctx_zeroshot,
                  out_path=str(out / "zeroshot.png"))
    plt.close("all")
    for a, nm in enumerate(asset_names):
        print(f"  {nm}: baseline={zs['baseline'][a]:.4f}  zs={zs['zeroshot'][a]:.4f}")

    print(f"\nAll outputs saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   default=None)
    parser.add_argument("--out_dir",  default=None)
    parser.add_argument("--n_train",  type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device:   cfg.device            = args.device
    if args.out_dir:  cfg.out_dir           = args.out_dir
    if args.n_train:  cfg.data.n_train_days = args.n_train
    if args.n_epochs: cfg.train.n_epochs    = args.n_epochs

    main(cfg)
