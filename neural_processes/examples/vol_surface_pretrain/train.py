"""
Transfer-learning training: pre-train on Heston, fine-tune on Group Tech.
Saves model_pretrained.pt, model_finetuned.pt, and both training curves.

Run from repo root:
    .venv/bin/python3 -m neural_processes.examples.vol_surface_pretrain.train
    .venv/bin/python3 -m neural_processes.examples.vol_surface_pretrain.train --device cpu
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

from .config import Config
from neural_processes.data import load_heston, load_grouptech
from neural_processes.models import Trainer
from neural_processes.models.cnp import MultiAssetCNP, transfer_weights
from .._shared import plot_training_curve


class _PhaseCfg:
    """Minimal config adapter so Trainer accepts the right model/train blocks."""
    def __init__(self, model, train, device, seed):
        self.model  = model
        self.train  = train
        self.device = device
        self.seed   = seed


def main(cfg: Config):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    # Phase 1: pre-train on Heston
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
    plot_training_curve(pre_history, out / "training_curves_pretrain.png",
                        title="Phase 1 — Heston pre-train (norm RMSE)")

    # Phase 2: transfer weights, fine-tune on Group Tech
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning on Group Tech market data")
    print("=" * 60)

    gt_ds = load_grouptech(
        cfg.data.csv_path,
        n_train_days=cfg.data.n_train_days,
        n_val_days=cfg.data.n_val_days,
        seed=cfg.seed,
        n_eval_days=cfg.data.n_eval_days,
    )
    print(f"  Train: {(gt_ds.split==0).sum()}  "
          f"Val: {(gt_ds.split==1).sum()}  "
          f"Assets: {gt_ds.n_assets}  "
          f"Points/day (max): {gt_ds.n_points}")

    mc        = cfg.model
    gt_module = MultiAssetCNP(
        n_assets=gt_ds.n_assets, q_dim=gt_ds.q_dim,
        d_asset=mc.d_asset,         d_model=mc.d_model,
        n_heads_obs=mc.n_heads_obs,     n_layers_obs=mc.n_layers_obs,
        n_heads_cross=mc.n_heads_cross, n_layers_cross=mc.n_layers_cross,
        d_latent=mc.d_latent,       d_hidden=mc.d_hidden,
        n_hidden_dec=mc.n_hidden_dec,   dropout=mc.dropout,
    )
    n_copied = transfer_weights(pretrained.module, gt_module)
    print(f"  Transferred {n_copied} weight tensors "
          f"(asset_embed and prior re-initialised for {gt_ds.n_assets} assets)")

    finetuned, ft_history = Trainer(cfg).train(gt_ds, init_module=gt_module)
    finetuned.save(str(out / "model_finetuned.pt"))
    plot_training_curve(ft_history, out / "training_curves.png",
                        title="Phase 2 — Group Tech fine-tune (norm RMSE)")

    print(f"\nModels and training artifacts saved to: {out}")
    return finetuned, gt_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device:  cfg.device  = args.device
    if args.out_dir: cfg.out_dir = args.out_dir

    main(cfg)
