"""
Analytics for the pretrain/fine-tune CNP experiment. Loads model_finetuned.pt from out_dir.

Run from repo root:
    .venv/bin/python3 -m neural_processes.examples.vol_surface_pretrain.analytics
    .venv/bin/python3 -m neural_processes.examples.vol_surface_pretrain.analytics --out_dir results/vol_surface_pretrain
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

from .config import Config, REPO_ROOT
from neural_processes.data import load_grouptech
import matplotlib.pyplot as plt
from neural_processes.models.cnp import FittedCNP
from neural_processes.analytics import (
    encode_dataset, pca_latent, plot_latent_trajectories, plot_ssr_evolution,
)
from .._shared import run_standard_analytics

RESULTS_CSV = str(REPO_ROOT / "results" / "experiment_results.csv")


def main(cfg: Config, model=None, dataset=None):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    if dataset is None:
        print("Loading data...")
        dataset = load_grouptech(
            cfg.data.csv_path,
            n_train_days=cfg.data.n_train_days,
            n_val_days=cfg.data.n_val_days,
            seed=cfg.seed,
            n_eval_days=cfg.data.n_eval_days,
        )
        print(f"  Assets: {dataset.n_assets}  Val days: {(dataset.split==1).sum()}")

    if model is None:
        model_path = out / "model_finetuned.pt"
        print(f"Loading model from {model_path}...")
        model = FittedCNP.load(str(model_path))

    print("\nAnalytics...")
    run_standard_analytics(model, dataset, cfg.analytics, out,
                           results_csv=RESULTS_CSV, experiment="vol_surface_pretrain")

    print("SSR evolution...")
    log_fwd = dataset.meta.get("log_fwd")
    log_fwd_val = log_fwd[val_idx] if log_fwd is not None else None
    plot_ssr_evolution(model, dataset, val_idx, log_fwd=log_fwd_val,
                       asset_names=asset_names,
                       out_path=str(out / "ssr_evolution.png"))
    plt.close("all")

    print("Latent trajectories...")
    asset_names = dataset.meta.get("asset_names",
                                   [f"asset_{i}" for i in range(dataset.n_assets)])
    val_idx = dataset.val_idx()
    z_val = encode_dataset(model, dataset, val_idx,
                           n_ctx=min(cfg.analytics.n_ctx_latent, dataset.ctx_max))
    z_2d, var_exp = pca_latent(z_val)
    plot_latent_trajectories(z_2d, var_exp, dataset.n_assets, asset_names,
                             out_path=str(out / "latent_trajectories.png"))
    plt.close("all")

    print(f"\nAll outputs saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.out_dir: cfg.out_dir = args.out_dir

    main(cfg)
