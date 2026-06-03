"""
Train a delta-CNP on Group Tech US hourly data.

Run from repo root:
    .venv/bin/python3 -m neural_processes.examples.vol_surface_increment_hourly.train
    .venv/bin/python3 -m neural_processes.examples.vol_surface_increment_hourly.train --device cpu --n_epochs 10
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

from .config import Config
from neural_processes.data import load_grouptech, compute_bspline_prior
from neural_processes.models import Trainer
from .._shared import plot_training_curve


def main(cfg: Config):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    print("Loading data (hourly)...")
    dataset = load_grouptech(
        cfg.data.csv_path,
        obs_col=cfg.data.obs_col,
        n_eval_days=cfg.data.n_eval_days,
        seed=cfg.seed,
    )
    print(f"  Train: {(dataset.split==0).sum()}  Val: {(dataset.split==1).sum()}"
          f"  Assets: {dataset.n_assets}  Points/obs (max): {dataset.n_points}")

    print("Computing BSpline priors (this may take a few minutes)...")
    dataset.prior_targets = compute_bspline_prior(dataset)
    n_valid = int(np.any(np.isfinite(dataset.prior_targets), axis=1).sum())
    print(f"  Prior available for {n_valid}/{dataset.n_days} observations")

    print("\nTraining...")
    model, history = Trainer(cfg).train(dataset)
    model.save(str(out / "model.pt"))
    plot_training_curve(history, out / "training_curve.png",
                        title="Training curve — increment model hourly (normalised delta RMSE)")
    print(f"\nModel and training artifacts saved to: {out}")
    return model, dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   default=None)
    parser.add_argument("--out_dir",  default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device:   cfg.device         = args.device
    if args.out_dir:  cfg.out_dir        = args.out_dir
    if args.n_epochs: cfg.train.n_epochs = args.n_epochs

    main(cfg)
