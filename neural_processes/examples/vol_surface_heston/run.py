"""
Base CNP experiment.

Run from repo root:
    python -m neural_processes.examples.vol_surface_heston.run
    python -m neural_processes.examples.vol_surface_heston.run --device cpu --n_train 100 --n_epochs 30
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import Config
from neural_processes.data import load_heston
from neural_processes.models import Trainer, FittedCNP
from neural_processes.analytics import (
    eval_rmse_vs_ctx, eval_per_feature_rmse, eval_zeroshot,
    plot_reconstruction, plot_rmse_vs_ctx, plot_per_feature_rmse, plot_zeroshot,
    encode_dataset, pca_latent,
    plot_pca_colored, plot_r2_heatmap, plot_latent_interpolation,
)


def main(cfg: Config):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    # 1. Data
    print("Loading data...")
    dataset, ood_ds = load_heston(
        cfg.data.train_path, cfg.data.ood_path,
        cfg.data.n_train_days, cfg.data.n_val_days, cfg.seed,
    )
    print(f"  Train: {(dataset.split==0).sum()}  Val: {(dataset.split==1).sum()}"
          + (f"  OOD: {ood_ds.n_days}" if ood_ds else ""))

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
    ood_rmse = None
    if ood_ds is not None:
        ood_idx  = np.arange(ood_ds.n_days)
        ood_rmse = eval_rmse_vs_ctx(model, ood_ds, ood_idx, ac.ctx_sizes_rmse_curve)
        last_nc  = ac.ctx_sizes_rmse_curve[-1]
        print(f"  OOD/Val ratio at n_ctx={last_nc}: "
              f"{ood_rmse[last_nc]/max(val_rmse[last_nc],1e-8):.2f}x")
    plot_rmse_vs_ctx(val_rmse, ood_rmse, out_path=str(out / "rmse_vs_ctx.png"))
    plt.close("all")
    with open(out / "rmse_vs_ctx.json", "w") as f:
        json.dump({"val": val_rmse, "ood": ood_rmse}, f, indent=2)

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

    print("Latent space analysis...")
    z_val  = encode_dataset(model, dataset, val_idx, n_ctx=min(ac.n_ctx_latent, dataset.ctx_max))
    z_2d, var_exp = pca_latent(z_val)
    print(f"  PC1 {var_exp[0]*100:.1f}%  PC2 {var_exp[1]*100:.1f}%")
    plot_pca_colored(z_2d, var_exp, dataset.params[val_idx],
                     ac.latent_param_names, ac.latent_param_indices, ac.latent_cmaps,
                     out_path=str(out / "latent_pca.png"))
    plot_r2_heatmap(z_2d, dataset.params[val_idx],
                    ac.latent_param_names, ac.latent_param_indices, asset_names,
                    out_path=str(out / "latent_r2.png"))
    plt.close("all")

    print("Latent interpolation...")
    v0_vals = dataset.params[val_idx, 0, 0]
    plot_latent_interpolation(
        model, dataset,
        day_lo=int(val_idx[np.argmin(v0_vals)]),
        day_hi=int(val_idx[np.argmax(v0_vals)]),
        out_path=str(out / "latent_interpolation.png"),
    )
    plt.close("all")

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
