"""Train the delta-CNP for the dissertation runs and cache it ONCE.

    .venv/bin/python -m surfacelab.experiments.train_thesis_cnp

The same weights are shared by both CNP variants in the thesis configs — the joint
(cross-asset attention) model and the per-asset (severed) ablation — so they must come
from a single training run.  Trains on the thesis split (last 900 days, train = first 800)
so the validation window the configs free-run over is never seen during training.
"""
from __future__ import annotations

import warnings
import torch

from surfacelab.data import compute_bspline_prior
from surfacelab.experiments.configs import (CNPTrainConfig, CNP_DELTA_CKPT, CNP_ABS_CKPT,
                                            _market_thesis)
from surfacelab.models import registry

warnings.filterwarnings("ignore")

EPOCHS = 20
BATCH = 32


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    print("Loading thesis market data (last 900 days, 800 train / 100 val) …")
    ds, _ = _market_thesis()()
    n_train = int((ds.split == 0).sum())
    n_val = int((ds.split == 1).sum())
    print(f"  {ds.n_days} days ({n_train} train / {n_val} val), "
          f"{ds.n_assets} assets, ctx_max={ds.ctx_max}")
    print("Computing B-spline prior (for the delta target) …")
    ds.prior_targets = compute_bspline_prior(ds)

    # absolute `cnp` (free-run / sequential) and `cnp_delta` (perfect-prior reference); each
    # backs both its joint and per-asset (nox) variant via shared weights.
    for name, ckpt in (("cnp", CNP_ABS_CKPT), ("cnp_delta", CNP_DELTA_CKPT)):
        cfg = CNPTrainConfig(device=dev)
        cfg.train.n_epochs = EPOCHS
        cfg.train.batch_size = BATCH
        model = registry.build(name, config=cfg, device=dev, checkpoint=ckpt)
        print(f"\n=== training {name} (batch={BATCH}, epochs={EPOCHS}) -> {ckpt} ===",
              flush=True)
        model.train(ds, saved=True, force=True)
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("\nDone. Thesis CNP variants will load these weights.")


if __name__ == "__main__":
    main()
