from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from neural_processes.data.base import SurfaceDataset
from .cnp import MultiAssetCNP, FittedCNP


class Trainer:
    """Trains a MultiAssetCNP on a SurfaceDataset and returns a FittedCNP."""

    def __init__(self, config):
        self.cfg = config

    def train(
        self,
        dataset: SurfaceDataset,
        init_module: MultiAssetCNP | None = None,
    ):
        """
        init_module: optional pre-initialised MultiAssetCNP (e.g. from transfer learning).
                     If provided it is used as-is; otherwise a fresh module is created.
        """
        cfg = self.cfg
        dev = torch.device(cfg.device)

        train_idx = dataset.train_idx()
        val_idx   = dataset.val_idx()
        n_assets  = dataset.n_assets
        ctx_max   = dataset.ctx_max

        # ── Feature normalisation (exclude zero-padded positions, T is dim 1) ──
        qf_train   = dataset.query_feats[train_idx]         # (N_tr, P, q_dim)
        valid_tr   = qf_train[:, :, 1] > 0                  # (N_tr, P)
        valid_flat = valid_tr.reshape(-1)
        feat_valid = qf_train.reshape(-1, dataset.q_dim)[valid_flat]
        feat_mean  = feat_valid.mean(0).astype(np.float32)
        feat_std   = (feat_valid.std(0) + 1e-8).astype(np.float32)

        # ── Per-asset log(IV) normalisation ───────────────────────────────────
        aids_train   = dataset.asset_ids[train_idx]
        log_iv_train = np.log(np.maximum(dataset.targets[train_idx], 1e-8))
        log_tgt_mean = np.zeros(n_assets, dtype=np.float32)
        log_tgt_std  = np.ones(n_assets,  dtype=np.float32)
        for a in range(n_assets):
            mask_a = valid_tr & (aids_train == a)
            vals   = log_iv_train[mask_a]
            if len(vals) > 1:
                log_tgt_mean[a] = float(vals.mean())
                log_tgt_std[a]  = float(vals.std() + 1e-8)

        # ── Normalise entire dataset ──────────────────────────────────────────
        feat_n = ((dataset.query_feats - feat_mean) / feat_std).astype(np.float32)
        log_iv = np.log(np.maximum(dataset.targets, 1e-8))
        tgt_n  = ((log_iv - log_tgt_mean[dataset.asset_ids]) /
                  log_tgt_std[dataset.asset_ids]).astype(np.float32)

        # ── Valid mask: T > 0 in raw query_feats means a real (non-padded) point ─
        valid_all = (dataset.query_feats[:, :, 1] > 0)      # (N_days, P)

        feat_t  = torch.from_numpy(feat_n).to(dev)
        tgt_t   = torch.from_numpy(tgt_n).to(dev)
        aid_t   = torch.from_numpy(dataset.asset_ids.astype(np.int64)).to(dev)
        valid_t = torch.from_numpy(valid_all).to(dev)

        mc = cfg.model
        if init_module is not None:
            module = init_module.to(dev)
        else:
            module = MultiAssetCNP(
                n_assets=n_assets, q_dim=dataset.q_dim,
                d_asset=mc.d_asset, d_model=mc.d_model,
                n_heads_obs=mc.n_heads_obs, n_layers_obs=mc.n_layers_obs,
                n_heads_cross=mc.n_heads_cross, n_layers_cross=mc.n_layers_cross,
                d_latent=mc.d_latent, d_hidden=mc.d_hidden,
                n_hidden_dec=mc.n_hidden_dec, dropout=mc.dropout,
            ).to(dev)

        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"Model: {n_params:,} trainable parameters")

        tc        = cfg.train
        optimizer = AdamW(module.parameters(), lr=tc.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=tc.n_epochs, eta_min=tc.lr / 20)

        train_rmse_hist, val_rmse_hist = [], []
        t0 = time.time()

        for epoch in range(1, tc.n_epochs + 1):
            module.train()
            perm = torch.randperm(len(train_idx), device=dev)
            total_sq, total_valid = 0.0, 0

            for b in range(0, len(train_idx), tc.batch_size):
                day_idx = torch.from_numpy(
                    train_idx[perm[b : b + tc.batch_size].cpu().numpy()]
                ).to(dev)
                n_ctx = int(np.random.randint(tc.ctx_min, ctx_max + 1))
                of, ot, oa, qf, qa, tgt, valid_tgt = _make_batch(
                    feat_t, tgt_t, aid_t, valid_t, day_idx, n_ctx, ctx_max, dev
                )
                pred  = module(of, ot, oa, qf, qa)
                # Fix 5: target pool only  |  Fix 1: exclude zero-padded positions
                err2  = (pred[:, ctx_max:] - tgt[:, ctx_max:]) ** 2
                denom = valid_tgt.sum().clamp(min=1)
                loss  = (err2 * valid_tgt).sum() / denom

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step()

                total_sq    += (err2 * valid_tgt).sum().item()
                total_valid += valid_tgt.sum().item()

            scheduler.step()
            train_rmse_hist.append(np.sqrt(total_sq / max(total_valid, 1)))

            if epoch % tc.log_every == 0:
                vr = _eval_rmse(module, feat_t, tgt_t, aid_t, valid_t,
                                val_idx, ctx_max, dev=dev)
                val_rmse_hist.append((epoch, vr))
                print(f"Epoch {epoch:4d}/{tc.n_epochs}  "
                      f"train {train_rmse_hist[-1]:.5f}  val {vr:.5f}  "
                      f"{time.time()-t0:.0f}s")

        print(f"Training done in {time.time()-t0:.1f}s")
        return (
            FittedCNP(module, feat_mean, feat_std, log_tgt_mean, log_tgt_std),
            {"train_rmse": train_rmse_hist, "val_rmse": val_rmse_hist},
        )


def _make_batch(feat_t, tgt_t, aid_t, valid_t, day_idx, n_ctx, ctx_max, dev):
    f = feat_t[day_idx]
    t = tgt_t[day_idx]
    a = aid_t[day_idx]
    v = valid_t[day_idx]
    perm     = torch.rand(len(day_idx), ctx_max, device=dev).argsort(dim=1)
    ctx_idx  = perm[:, :n_ctx]
    exp_idx  = ctx_idx.unsqueeze(-1).expand(-1, -1, f.shape[-1])
    obs_feat = f[:, :ctx_max].gather(1, exp_idx)
    obs_tgt  = t[:, :ctx_max].gather(1, ctx_idx)
    obs_aid  = a[:, :ctx_max].gather(1, ctx_idx)
    valid_tgt = v[:, ctx_max:].float()          # (B, target_pool) as float for weighting
    return obs_feat, obs_tgt, obs_aid, f, a, t, valid_tgt


@torch.no_grad()
def _eval_rmse(module, feat_t, tgt_t, aid_t, valid_t, val_idx, ctx_max, dev,
               n_ctx=30, n_reps=4):
    module.eval()
    total_sq, total_valid = 0.0, 0
    val_t = torch.from_numpy(val_idx).to(dev)
    for _ in range(n_reps):
        of, ot, oa, qf, qa, tgt, valid_tgt = _make_batch(
            feat_t, tgt_t, aid_t, valid_t, val_t, n_ctx, ctx_max, dev
        )
        pred = module(of, ot, oa, qf, qa)
        err2  = (pred[:, ctx_max:] - tgt[:, ctx_max:]) ** 2
        total_sq    += (err2 * valid_tgt).sum().item()
        total_valid += valid_tgt.sum().item()
    return np.sqrt(total_sq / max(total_valid, 1))
