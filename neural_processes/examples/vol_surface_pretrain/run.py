"""
Transfer-learning experiment: pre-train on Heston synthetic data, fine-tune on
Group Tech market data.

Run from repo root:
    python -m neural_processes.examples.vol_surface_pretrain.run
    python -m neural_processes.examples.vol_surface_pretrain.run --device cpu
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import Config
from neural_processes.data import load_heston, load_grouptech
from neural_processes.models import Trainer
from neural_processes.models.cnp import MultiAssetCNP, transfer_weights
from neural_processes.analytics import (
    eval_rmse_vs_ctx, eval_per_feature_rmse, eval_zeroshot,
    plot_reconstruction, plot_rmse_vs_ctx, plot_per_feature_rmse, plot_zeroshot,
)


class _PhaseCfg:
    """Minimal config shim so Trainer accepts the right model/train blocks."""
    def __init__(self, model, train, device, seed):
        self.model  = model
        self.train  = train
        self.device = device
        self.seed   = seed


def main(cfg: Config):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    # ══ Phase 1: pre-train on Heston ═════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: Pre-training on Heston synthetic data")
    print("=" * 60)

    heston_ds, _ = load_heston(
        cfg.pretrain_data.train_path,
        cfg.pretrain_data.ood_path,
        cfg.pretrain_data.n_train_days,
        cfg.pretrain_data.n_val_days,
        cfg.seed,
    )
    print(f"  Train: {(heston_ds.split==0).sum()}  "
          f"Val: {(heston_ds.split==1).sum()}  "
          f"Assets: {heston_ds.n_assets}")

    pretrained, pre_history = Trainer(
        _PhaseCfg(cfg.model, cfg.pretrain_train, cfg.device, cfg.seed)
    ).train(heston_ds)
    pretrained.save(str(out / "model_pretrained.pt"))

    # ══ Phase 2: transfer weights, fine-tune on Group Tech ════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning on Group Tech market data")
    print("=" * 60)

    gt_ds = load_grouptech(
        cfg.data.csv_path,
        n_train_days=cfg.data.n_train_days,
        n_val_days=cfg.data.n_val_days,
        seed=cfg.seed,
        val_frac=cfg.data.val_frac,
    )
    print(f"  Train: {(gt_ds.split==0).sum()}  "
          f"Val: {(gt_ds.split==1).sum()}  "
          f"Assets: {gt_ds.n_assets}  "
          f"Points/day (max): {gt_ds.n_points}")

    mc        = cfg.model
    gt_module = MultiAssetCNP(
        n_assets=gt_ds.n_assets, q_dim=gt_ds.q_dim,
        d_asset=mc.d_asset,       d_model=mc.d_model,
        n_heads_obs=mc.n_heads_obs,   n_layers_obs=mc.n_layers_obs,
        n_heads_cross=mc.n_heads_cross, n_layers_cross=mc.n_layers_cross,
        d_latent=mc.d_latent,     d_hidden=mc.d_hidden,
        n_hidden_dec=mc.n_hidden_dec, dropout=mc.dropout,
    )
    n_copied = transfer_weights(pretrained.module, gt_module)
    print(f"  Transferred {n_copied} weight tensors "
          f"(asset_embed and prior re-initialised for {gt_ds.n_assets} assets)")

    finetuned, ft_history = Trainer(cfg).train(gt_ds, init_module=gt_module)
    finetuned.save(str(out / "model_finetuned.pt"))

    # ══ Training curves ════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, hist, title in zip(
        axes,
        [pre_history, ft_history],
        ["Phase 1 — Heston pre-train (norm RMSE)", "Phase 2 — Group Tech fine-tune (norm RMSE)"],
    ):
        ax.plot(hist["train_rmse"], lw=1, alpha=0.8, label="Train")
        if hist["val_rmse"]:
            vx, vy = zip(*hist["val_rmse"])
            ax.plot(vx, vy, "o-", color="C1", ms=4, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE (norm)"); ax.set_yscale("log")
        ax.legend(); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=120)
    plt.close(fig)

    # ══ Analytics on GroupTech val set ════════════════════════════════════════
    model       = finetuned
    ac          = cfg.analytics
    val_idx     = gt_ds.val_idx()
    n_assets    = gt_ds.n_assets
    asset_names = gt_ds.meta.get("asset_names", [f"asset_{i}" for i in range(n_assets)])
    feat_names  = gt_ds.meta.get("query_feat_names", ["x", "group"])

    print("\nReconstruction plot...")
    plot_reconstruction(model, gt_ds, val_idx[0], ctx_sizes=ac.ctx_sizes_recon,
                        out_path=str(out / "reconstruction.png"))
    plt.close("all")

    print("RMSE vs context size...")
    val_rmse = eval_rmse_vs_ctx(model, gt_ds, val_idx, ac.ctx_sizes_rmse_curve)
    plot_rmse_vs_ctx(val_rmse, ood_rmse=None, out_path=str(out / "rmse_vs_ctx.png"))
    plt.close("all")
    with open(out / "rmse_vs_ctx.json", "w") as f:
        json.dump({"val": val_rmse}, f, indent=2)

    print("Per-maturity RMSE...")
    mat_rmse = eval_per_feature_rmse(model, gt_ds, val_idx,
                                      n_ctx=ac.n_ctx_per_maturity, feat_dim=1)
    plot_per_feature_rmse(
        mat_rmse,
        feat_name=feat_names[1] if len(feat_names) > 1 else "maturity",
        n_ctx=ac.n_ctx_per_maturity,
        out_path=str(out / "per_maturity_rmse.png"),
    )
    plt.close("all")

    print("Zero-shot test...")
    zs = eval_zeroshot(model, gt_ds, val_idx[:50], n_ctx=ac.n_ctx_zeroshot)
    plot_zeroshot(zs, asset_names=asset_names, n_ctx=ac.n_ctx_zeroshot,
                  out_path=str(out / "zeroshot.png"))
    plt.close("all")
    for a, nm in enumerate(asset_names):
        print(f"  {nm}: baseline={zs['baseline'][a]:.4f}  zs={zs['zeroshot'][a]:.4f}")

    print(f"\nAll outputs saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device:  cfg.device  = args.device
    if args.out_dir: cfg.out_dir = args.out_dir

    main(cfg)
