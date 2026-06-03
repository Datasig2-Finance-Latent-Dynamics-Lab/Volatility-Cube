"""
Analytics for the Heston CNP experiment. Loads model.pt from out_dir.

Includes standard surface analytics plus Heston-specific latent space analysis
(PCA coloured by ground-truth Heston parameters, R² heatmap, latent interpolation).

Run from repo root:
    .venv/bin/python3 -m neural_processes.examples.vol_surface_heston.analytics
    .venv/bin/python3 -m neural_processes.examples.vol_surface_heston.analytics --out_dir results/vol_surface_neural
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import Config, REPO_ROOT
from neural_processes.data import load_heston
from neural_processes.models.cnp import FittedCNP
from neural_processes.analytics import (
    encode_dataset, pca_latent,
    plot_pca_colored, plot_r2_heatmap, plot_latent_interpolation,
    plot_latent_trajectories, plot_ssr_evolution,
)
from .._shared import run_standard_analytics

RESULTS_CSV = str(REPO_ROOT / "results" / "experiment_results.csv")


def main(cfg: Config, model=None, dataset=None, ood_ds=None):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    if dataset is None:
        print("Loading data...")
        dataset, ood_ds = load_heston(
            cfg.data.train_path, cfg.data.ood_path,
            cfg.data.n_train_days, cfg.data.n_val_days, cfg.seed,
        )
        print(f"  Val: {(dataset.split==1).sum()}"
              + (f"  OOD: {ood_ds.n_days}" if ood_ds else ""))

    if model is None:
        model_path = out / "model.pt"
        print(f"Loading model from {model_path}...")
        model = FittedCNP.load(str(model_path))

    print("\nAnalytics...")
    run_standard_analytics(model, dataset, cfg.analytics, out, ood_ds=ood_ds,
                           results_csv=RESULTS_CSV, experiment="vol_surface_heston")

    # Latent space analysis (Heston-specific: ground-truth params available)
    ac = cfg.analytics
    val_idx = dataset.val_idx()
    asset_names = dataset.meta.get("asset_names",
                                   [f"asset_{i}" for i in range(dataset.n_assets)])

    print("SSR evolution (ATM vol + skew — no forward prices for Heston)...")
    plot_ssr_evolution(model, dataset, val_idx, log_fwd=None,
                       asset_names=asset_names,
                       out_path=str(out / "ssr_evolution.png"))
    plt.close("all")

    print("Latent space analysis...")
    z_val = encode_dataset(model, dataset, val_idx, n_ctx=min(ac.n_ctx_latent, dataset.ctx_max))
    z_2d, var_exp = pca_latent(z_val)
    print(f"  PC1 {var_exp[0]*100:.1f}%  PC2 {var_exp[1]*100:.1f}%")
    plot_pca_colored(z_2d, var_exp, dataset.params[val_idx],
                     ac.latent_param_names, ac.latent_param_indices, ac.latent_cmaps,
                     out_path=str(out / "latent_pca.png"))
    plot_r2_heatmap(z_2d, dataset.params[val_idx],
                    ac.latent_param_names, ac.latent_param_indices, asset_names,
                    out_path=str(out / "latent_r2.png"))
    plot_latent_trajectories(z_2d, var_exp, dataset.n_assets, asset_names,
                             out_path=str(out / "latent_trajectories.png"))
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
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.out_dir: cfg.out_dir = args.out_dir

    main(cfg)
