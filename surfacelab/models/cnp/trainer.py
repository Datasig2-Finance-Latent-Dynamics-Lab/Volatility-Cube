"""CNP training loop — trains a MultiAssetCNP on a Dataset, returns a Fitted(Delta)CNP."""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import norm

from surfacelab.data.dataset import Dataset as SurfaceDataset
from surfacelab.models.cnp.module import MultiAssetCNP, FittedCNP, FittedDeltaCNP


def _spread_weights(dataset: SurfaceDataset, strength: float = 1.0,
                    clamp=(0.1, 10.0)) -> np.ndarray:
    """Per-point loss weights ∝ (median_spread / IV-spread)^(2·strength), normalised to mean 1.

    The stored bid/ask are forward-normalised call prices, so a price half-spread maps to an
    IV tolerance via vega: IV_spread ≈ price_spread / (√T · φ(d1)).  Tilting the squared IV
    error by an inverse-spread weight makes the objective lean toward the "is the predicted
    price inside the quoted bid-ask spread" diagnostic — tight liquid ATM quotes pull harder,
    wide illiquid wings count less.

    `strength` controls how hard that tilt bites:
      * 0.0  → every (valid) point gets weight 1  → plain unweighted RMSE.
      * 1.0  → full 1/IV-spread² weighting (the original heavy tilt; ATM dominates, wings
               are almost ignored — which is why raw vol-point RMSE in the wings stays high).
      * (0,1) → a gentle in-between (use a tight `clamp` so it stays light).
    The point with the median spread always gets weight ≈1.  Returns 0 for invalid/padding.
    """
    k = dataset.query_feats[:, :, 0]
    T = dataset.query_feats[:, :, 1]
    iv = dataset.targets
    valid = ((T > 0) & np.isfinite(dataset.bid) & np.isfinite(dataset.ask)
             & np.isfinite(iv) & (iv > 0) & ((dataset.ask - dataset.bid) > 0))
    if not valid.any() or strength <= 0:
        return np.ones_like(iv, dtype=np.float32)
    sqT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (-k + 0.5 * iv ** 2 * T) / (iv * sqT + 1e-14)
    vega = sqT * norm.pdf(d1)                                  # normalised BS vega
    iv_spread = (dataset.ask - dataset.bid) / np.maximum(vega, 1e-8)
    med = np.median(iv_spread[valid])                          # median point -> weight 1
    r = (med / np.maximum(iv_spread, 1e-12)) ** 2              # tight < median: >1; wide: <1
    w = r ** float(strength)                                   # strength dials the tilt
    w = np.clip(w, clamp[0], clamp[1])
    w = np.where(valid, w, 0.0)
    w = w / max(w[valid].mean(), 1e-8)                         # mean over valid -> 1
    return w.astype(np.float32)


# Loss-weighting presets: mode -> (inverse-spread strength, clamp). "none" is plain
# unweighted RMSE; "light" is a gentle tilt with a tight clamp; "spread" is the original
# heavy 1/IV-spread² weighting that lets tight ATM quotes dominate the objective.
_WEIGHT_MODES = {
    "none":   (0.0, (1.0, 1.0)),
    "light":  (0.35, (0.5, 2.0)),
    "spread": (1.0, (0.1, 10.0)),
}


def _resolve_weighting(train_cfg):
    """Resolve the loss-weighting mode + strength + clamp from a train config.

    Prefers the new `loss_weighting` string knob ("none"|"light"|"spread"); falls back to
    the legacy `spread_weighted` bool (True->"spread", False->"none") for old configs.
    An explicit `loss_weight_strength` (float) overrides the preset strength if present.
    """
    mode = getattr(train_cfg, "loss_weighting", None)
    if mode is None:
        mode = "spread" if getattr(train_cfg, "spread_weighted", True) else "none"
    if mode not in _WEIGHT_MODES:
        raise ValueError(f"loss_weighting must be one of {sorted(_WEIGHT_MODES)}, got {mode!r}")
    strength, clamp = _WEIGHT_MODES[mode]
    override = getattr(train_cfg, "loss_weight_strength", None)
    if override is not None:
        strength = float(override)
    return mode, strength, clamp


class Trainer:
    """Trains a MultiAssetCNP on a Dataset and returns a Fitted(Delta)CNP."""

    def __init__(self, config):
        self.cfg = config

    def train(self, dataset: SurfaceDataset, init_module: MultiAssetCNP | None = None):
        cfg = self.cfg
        dev = torch.device(cfg.device)

        train_idx = dataset.train_idx()
        val_idx = dataset.val_idx()
        n_assets = dataset.n_assets
        ctx_max = dataset.ctx_max
        delta_mode = dataset.prior_targets is not None

        if delta_mode:
            has_prior = np.array([np.any(np.isfinite(dataset.prior_targets[i]))
                                  for i in range(dataset.n_days)])
            train_idx = train_idx[has_prior[train_idx]]
            val_idx = val_idx[has_prior[val_idx]]

        qf_train = dataset.query_feats[train_idx]
        valid_tr = qf_train[:, :, 1] > 0
        feat_valid = qf_train.reshape(-1, dataset.q_dim)[valid_tr.reshape(-1)]
        feat_mean = feat_valid.mean(0).astype(np.float32)
        feat_std = (feat_valid.std(0) + 1e-8).astype(np.float32)

        if delta_mode:
            aids_train = dataset.asset_ids[train_idx]
            delta_train = dataset.targets[train_idx] - dataset.prior_targets[train_idx]
            delta_mean = np.zeros(n_assets, dtype=np.float32)
            delta_std = np.ones(n_assets, dtype=np.float32)
            for a in range(n_assets):
                vals = delta_train[valid_tr & (aids_train == a)]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 1:
                    lo, hi = np.percentile(vals, [1.0, 99.0])
                    vals_r = vals[(vals >= lo) & (vals <= hi)]
                    use = vals_r if len(vals_r) > 1 else vals
                    delta_mean[a] = float(use.mean())
                    delta_std[a] = float(use.std() + 1e-8)
            delta_all = dataset.targets - dataset.prior_targets
            tgt_n = ((delta_all - delta_mean[dataset.asset_ids])
                     / delta_std[dataset.asset_ids]).astype(np.float32)
            tgt_n = np.where(np.isfinite(tgt_n), tgt_n, 0.0).astype(np.float32)
            valid_all = (dataset.query_feats[:, :, 1] > 0) & np.isfinite(dataset.prior_targets)
        else:
            aids_train = dataset.asset_ids[train_idx]
            log_iv_train = np.log(np.maximum(dataset.targets[train_idx], 1e-8))
            log_tgt_mean = np.zeros(n_assets, dtype=np.float32)
            log_tgt_std = np.ones(n_assets, dtype=np.float32)
            for a in range(n_assets):
                vals = log_iv_train[valid_tr & (aids_train == a)]
                if len(vals) > 1:
                    log_tgt_mean[a] = float(vals.mean())
                    log_tgt_std[a] = float(vals.std() + 1e-8)
            log_iv = np.log(np.maximum(dataset.targets, 1e-8))
            tgt_n = ((log_iv - log_tgt_mean[dataset.asset_ids])
                     / log_tgt_std[dataset.asset_ids]).astype(np.float32)
            valid_all = (dataset.query_feats[:, :, 1] > 0)

        feat_n = ((dataset.query_feats - feat_mean) / feat_std).astype(np.float32)
        feat_t = torch.from_numpy(feat_n).to(dev)
        tgt_t = torch.from_numpy(tgt_n).to(dev)
        aid_t = torch.from_numpy(dataset.asset_ids.astype(np.int64)).to(dev)
        valid_t = torch.from_numpy(valid_all).to(dev)

        # Per-point loss weights.  Default is "none" → plain unweighted RMSE: every valid
        # point counts equally, so the model is forced to fit the illiquid wings too (where
        # vol-point errors are largest) instead of coasting on tight ATM quotes.  "light"
        # applies a gentle inverse-spread tilt; "spread" is the old heavy 1/IV-spread²
        # weighting.  Tilts need bid/ask, so non-market data (e.g. Heston) is always uniform.
        mode, strength, clamp = _resolve_weighting(cfg.train)
        if mode != "none" and dataset.bid is not None and dataset.ask is not None:
            w_np = _spread_weights(dataset, strength=strength, clamp=clamp)
            print(f"Loss weighting '{mode}' (strength={strength}): weights ∈ "
                  f"[{w_np[w_np > 0].min():.2f}, {w_np.max():.2f}], mean(valid)=1.0")
        else:
            if mode != "none":
                print(f"Loss weighting '{mode}' requested but no bid/ask — using unweighted RMSE.")
            else:
                print("Loss weighting 'none' — plain unweighted RMSE.")
            w_np = np.ones_like(dataset.targets, dtype=np.float32)
        w_t = torch.from_numpy(w_np).to(dev)

        mc = cfg.model
        if init_module is not None:
            module = init_module.to(dev)
        else:
            module = MultiAssetCNP(
                n_assets=n_assets, q_dim=dataset.q_dim,
                d_asset=mc.d_asset, d_model=mc.d_model, n_heads=mc.n_heads,
                n_layers_enc=mc.n_layers_enc, n_layers_dec=mc.n_layers_dec,
                d_hidden=mc.d_hidden, n_fourier=mc.n_fourier,
                fourier_scale=mc.fourier_scale, dropout=mc.dropout).to(dev)

        print(f"Model: {sum(p.numel() for p in module.parameters()):,} params")
        tc = cfg.train
        optimizer = AdamW(module.parameters(), lr=tc.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=tc.n_epochs, eta_min=tc.lr / 20)
        train_hist, val_hist = [], []
        t0 = time.time()

        for epoch in range(1, tc.n_epochs + 1):
            module.train()
            perm = torch.randperm(len(train_idx), device=dev)
            total_sq = total_valid = 0
            for b in range(0, len(train_idx), tc.batch_size):
                day_idx = torch.from_numpy(
                    train_idx[perm[b:b + tc.batch_size].cpu().numpy()]).to(dev)
                n_ctx = int(np.random.randint(tc.ctx_min, ctx_max + 1))
                of, ot, oa, ov, qf, qa, tgt, valid_tgt, w_tgt = _make_batch(
                    feat_t, tgt_t, aid_t, valid_t, day_idx, n_ctx, dev, w_t)
                pred = module(of, ot, oa, qf, qa, obs_valid=ov)
                err2 = (pred - tgt) ** 2
                wm = valid_tgt * w_tgt
                loss = (err2 * wm).sum() / wm.sum().clamp(min=1)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step()
                total_sq += (err2 * wm).sum().item()
                total_valid += wm.sum().item()
            scheduler.step()
            train_hist.append(np.sqrt(total_sq / max(total_valid, 1)))
            if epoch % tc.log_every == 0:
                vr = _eval_rmse(module, feat_t, tgt_t, aid_t, valid_t, val_idx, dev, w_t)
                val_hist.append((epoch, vr))
                print(f"Epoch {epoch:4d}/{tc.n_epochs}  train {train_hist[-1]:.5f}  "
                      f"val {vr:.5f}  {time.time()-t0:.0f}s", flush=True)
            else:
                print(f"Epoch {epoch:4d}/{tc.n_epochs}  train {train_hist[-1]:.5f}  "
                      f"{time.time()-t0:.0f}s", flush=True)

        print(f"Training done in {time.time()-t0:.1f}s")
        if delta_mode:
            fitted = FittedDeltaCNP(module, feat_mean, feat_std, delta_mean, delta_std)
        else:
            fitted = FittedCNP(module, feat_mean, feat_std, log_tgt_mean, log_tgt_std)
        return fitted, {"train_rmse": train_hist, "val_rmse": val_hist}


def _make_batch(feat_t, tgt_t, aid_t, valid_t, day_idx, n_ctx, dev, w_t):
    """A CNP training batch matching the eval's uniform regime.

    Context = a random n_ctx-subset of the day's VALID points, sampled uniformly across the
    WHOLE surface (NOT the first ctx_max positions — those are sorted to a single maturity
    band, which would train the model to only extrapolate short→long and never use context).
    Targets = the full surface (every valid point), so the loss rewards both reproducing the
    observed context and interpolating to the rest — exactly what the eval scores.
    """
    f, t, a, v = feat_t[day_idx], tgt_t[day_idx], aid_t[day_idx], valid_t[day_idx]
    w = w_t[day_idx]
    B, P, _ = f.shape
    # uniform sample of n_ctx valid positions per day: random scores, invalid pushed last.
    score = torch.rand(B, P, device=dev).masked_fill(~v, float("inf"))
    ctx_idx = score.argsort(dim=1)[:, :n_ctx]
    exp_idx = ctx_idx.unsqueeze(-1).expand(-1, -1, f.shape[-1])
    obs_feat = f.gather(1, exp_idx)
    obs_tgt = t.gather(1, ctx_idx)
    obs_aid = a.gather(1, ctx_idx)
    obs_valid = v.gather(1, ctx_idx)        # False where a day had < n_ctx valid points
    valid_tgt = v.float()                   # score the whole surface
    w_tgt = w.float()
    # query = the full-day arrays (f, a, t), unpacked at call sites as (qf, qa, tgt)
    return obs_feat, obs_tgt, obs_aid, obs_valid, f, a, t, valid_tgt, w_tgt


@torch.no_grad()
def _eval_rmse(module, feat_t, tgt_t, aid_t, valid_t, val_idx, dev, w_t,
               n_ctx=30, n_reps=4):
    """Plain (unweighted) held-out RMSE — a training diagnostic, not the weighted objective."""
    module.eval()
    total_sq = total_valid = 0
    val_t = torch.from_numpy(val_idx).to(dev)
    for _ in range(n_reps):
        of, ot, oa, ov, qf, qa, tgt, valid_tgt, _w_tgt = _make_batch(
            feat_t, tgt_t, aid_t, valid_t, val_t, n_ctx, dev, w_t)
        pred = module(of, ot, oa, qf, qa, obs_valid=ov)
        err2 = (pred - tgt) ** 2
        total_sq += (err2 * valid_tgt).sum().item()
        total_valid += valid_tgt.sum().item()
    return np.sqrt(total_sq / max(total_valid, 1))
