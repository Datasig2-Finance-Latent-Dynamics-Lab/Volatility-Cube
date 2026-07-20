"""Train the market CNP variants and cache them under trained_models/.

    .venv/bin/python -m surfacelab.experiments.train_market_cnps

Loss weighting is set by CNPTrainConfig.train.loss_weighting (default "none" = plain
unweighted RMSE); see configs.py for "light"/"spread" alternatives.

Variants:
  cnp        — absolute IV, mean-pool encoder
  cnp_delta  — increment off the B-spline prior, mean-pool encoder
"""
from __future__ import annotations

import warnings
import torch

from surfacelab.data import load_grouptech, compute_bspline_prior
from surfacelab.experiments.configs import MARKET_CSV, CNPTrainConfig, TRAINED
from surfacelab.models import registry

warnings.filterwarnings("ignore")

EPOCHS = 20
BASE_BATCH = 32


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    print("Loading market data …")
    ds = load_grouptech(MARKET_CSV, n_eval_days=30)
    print(f"  {ds.n_days} days, {ds.n_assets} assets, ctx_max={ds.ctx_max}")
    print("Computing B-spline prior (for the delta variant) …")
    ds.prior_targets = compute_bspline_prior(ds)

    # (registry name, batch size) — delta vs absolute is encoded in the registry name.
    specs = [
        ("cnp",       BASE_BATCH),
        ("cnp_delta", BASE_BATCH),
    ]
    for name, batch in specs:
        cfg = CNPTrainConfig(device=dev)
        cfg.train.n_epochs = EPOCHS
        cfg.train.batch_size = batch
        ckpt = str(TRAINED / f"{name}_market.pt")
        model = registry.build(name, config=cfg, device=dev, checkpoint=ckpt)
        print(f"\n=== training {name}  (batch={batch}, epochs={EPOCHS}) -> {ckpt} ===",
              flush=True)
        model.train(ds, saved=True, force=True)
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("\nAll market CNP variants trained and cached.")


if __name__ == "__main__":
    main()
