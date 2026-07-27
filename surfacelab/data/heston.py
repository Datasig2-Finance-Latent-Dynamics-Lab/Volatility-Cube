from __future__ import annotations
import numpy as np
from pathlib import Path
from surfacelab.data.dataset import Dataset as SurfaceDataset


def load_heston(
    train_path: str,
    ood_path: str | None = None,
    n_train_days: int | None = None,
    n_val_days: int | None = None,
    seed: int = 0,
) -> tuple["SurfaceDataset", "SurfaceDataset | None"]:
    """Load Heston multi-asset dataset(s) from .npz files produced by generate_heston.

    Parameters
    ----------
    train_path    : path to heston_multiasset_training.npz
    ood_path      : optional path to heston_multiasset_ood_test.npz
    n_train_days  : subsample this many training days (None = use all)
    n_val_days    : subsample this many validation days (None = use all)
    seed          : RNG seed for subsampling
    """
    raw = np.load(train_path)
    lm      = raw["lm"]
    T       = raw["maturity"]
    aid     = raw["asset_id"]
    iv      = raw["iv"]
    split   = raw["split"]
    params  = raw["params"]
    ctx_max = int(raw["ctx_max"])

    rng = np.random.default_rng(seed)
    train_idx_all = np.where(split == 0)[0]
    val_idx_all   = np.where(split == 1)[0]

    if n_train_days is not None and n_train_days < len(train_idx_all):
        train_idx_all = np.sort(rng.choice(train_idx_all, n_train_days, replace=False))
    if n_val_days is not None and n_val_days < len(val_idx_all):
        val_idx_all = np.sort(rng.choice(val_idx_all, n_val_days, replace=False))

    keep      = np.sort(np.concatenate([train_idx_all, val_idx_all]))
    new_split = np.where(np.isin(keep, val_idx_all), 1, 0).astype(np.int8)

    query_feats = np.stack([lm[keep], T[keep]], axis=-1).astype(np.float32)
    n_assets    = int(params.shape[1])

    ds = SurfaceDataset(
        query_feats=query_feats,
        asset_ids=aid[keep].astype(np.int64),
        targets=iv[keep].astype(np.float32),
        split=new_split,
        ctx_max=ctx_max,
        n_assets=n_assets,
        params=params[keep].astype(np.float32),
        meta={
            "query_feat_names": ["lm", "T"],
            "target_name": "IV",
            "dgp": "heston_multiasset",
        },
    )

    ood_ds = None
    if ood_path is not None and Path(ood_path).exists():
        raw_ood = np.load(ood_path)
        qf_ood  = np.stack([raw_ood["lm"], raw_ood["maturity"]], axis=-1).astype(np.float32)
        n_ood   = qf_ood.shape[0]
        ood_ds  = SurfaceDataset(
            query_feats=qf_ood,
            asset_ids=raw_ood["asset_id"].astype(np.int64),
            targets=raw_ood["iv"].astype(np.float32),
            split=np.zeros(n_ood, dtype=np.int8),
            ctx_max=int(raw_ood["ctx_max"]),
            n_assets=int(raw_ood["params"].shape[1]),
            params=raw_ood["params"].astype(np.float32),
            meta={"query_feat_names": ["lm", "T"], "target_name": "IV",
                  "dgp": "heston_multiasset_ood"},
        )

    return ds, ood_ds
