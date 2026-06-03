"""
Train CNP on Heston synthetic data. Saves model.pt and training_curve.png.

Run from repo root:
    .venv/bin/python3 -m neural_processes.examples.vol_surface_heston.train
    .venv/bin/python3 -m neural_processes.examples.vol_surface_heston.train --device cpu --n_train 100 --n_epochs 30
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

from .config import Config
from neural_processes.data import load_heston
from neural_processes.models import Trainer
from .._shared import plot_training_curve


def main(cfg: Config):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    print("Loading data...")
    dataset, ood_ds = load_heston(
        cfg.data.train_path, cfg.data.ood_path,
        cfg.data.n_train_days, cfg.data.n_val_days, cfg.seed,
    )
    print(f"  Train: {(dataset.split==0).sum()}  Val: {(dataset.split==1).sum()}"
          + (f"  OOD: {ood_ds.n_days}" if ood_ds else ""))

    print("\nTraining...")
    model, history = Trainer(cfg).train(dataset)
    model.save(str(out / "model.pt"))
    plot_training_curve(history, out / "training_curve.png")
    print(f"\nModel and training artifacts saved to: {out}")
    return model, dataset, ood_ds


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
