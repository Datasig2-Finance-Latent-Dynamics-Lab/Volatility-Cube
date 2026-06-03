from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .base import SurfaceModel


class MultiAssetCNP(nn.Module):
    """
    Conditional Neural Process for multi-asset vol surfaces.

    Encoder:
      1. ObsTransformer  — all context observations (all assets mixed) attend to each other
      2. Per-asset mean pool → CrossAssetTransformer → per-asset latent codes

    Decoder: z_asset || query_feats → MLP → target
    """

    def __init__(
        self,
        n_assets: int,
        q_dim: int,
        d_asset: int = 8,
        d_model: int = 32,
        n_heads_obs: int = 4,
        n_layers_obs: int = 2,
        n_heads_cross: int = 4,
        n_layers_cross: int = 3,
        d_latent: int = 16,
        d_hidden: int = 128,
        n_hidden_dec: int = 3,
        dropout: float = 0.05,
    ):
        """TODO.

        Args:
            n_assets: TODO.
            q_dim: TODO.
            d_asset: TODO.
            d_model: TODO.
            n_heads_obs: TODO.
            n_layers_obs: TODO.
            n_heads_cross: TODO.
            n_layers_cross: TODO.
            d_latent: TODO.
            d_hidden: TODO.
            n_hidden_dec: TODO.
            dropout: TODO.
        """
        super().__init__()
        self.n_assets = n_assets
        self.d_latent = d_latent

        self.asset_embed = nn.Embedding(n_assets, d_asset)
        self.obs_proj    = nn.Linear(d_asset + q_dim + 1, d_model)  # +1 for target

        obs_layer = nn.TransformerEncoderLayer(
            d_model, n_heads_obs, d_model * 4, dropout, batch_first=True, norm_first=True)
        self.obs_enc  = nn.TransformerEncoder(obs_layer, n_layers_obs)
        self.obs_norm = nn.LayerNorm(d_model)

        self.prior = nn.Parameter(torch.zeros(n_assets, d_model))

        cross_layer = nn.TransformerEncoderLayer(
            d_model, n_heads_cross, d_model * 4, dropout, batch_first=True, norm_first=True)
        self.cross_enc  = nn.TransformerEncoder(cross_layer, n_layers_cross)
        self.cross_norm = nn.LayerNorm(d_model)
        self.to_latent  = nn.Linear(d_model, d_latent)

        layers = [nn.Linear(d_latent + q_dim, d_hidden), nn.GELU()]
        for _ in range(n_hidden_dec - 1):
            layers += [nn.Dropout(dropout), nn.Linear(d_hidden, d_hidden), nn.GELU()]
        layers += [nn.Linear(d_hidden, 1)]
        self.decoder = nn.Sequential(*layers)

    def encode(self, obs_feat, obs_tgt, obs_aid):
        """TODO.

        Args:
            obs_feat: TODO.
            obs_tgt: TODO.
            obs_aid: TODO.

        Returns:
            TODO.
        """
        B = obs_feat.shape[0]
        e = self.asset_embed(obs_aid)
        x = torch.cat([e, obs_feat, obs_tgt.unsqueeze(-1)], dim=-1)
        h = self.obs_norm(self.obs_enc(self.obs_proj(x)))

        one_hot  = F.one_hot(obs_aid, self.n_assets).float()
        sums     = torch.einsum("bcp,bcd->bpd", one_hot, h)
        counts   = one_hot.sum(1).unsqueeze(-1).clamp(min=1)
        means    = sums / counts
        has_data = counts.gt(0)
        slots    = torch.where(has_data, means, self.prior.unsqueeze(0).expand(B, -1, -1))

        slots = self.cross_norm(self.cross_enc(slots))
        return self.to_latent(slots)

    def decode(self, z, qry_feat, qry_aid):
        """TODO.

        Args:
            z: TODO.
            qry_feat: TODO.
            qry_aid: TODO.

        Returns:
            TODO.
        """
        B = qry_aid.shape[0]
        z_q = z[torch.arange(B, device=z.device).unsqueeze(1), qry_aid]
        return self.decoder(torch.cat([z_q, qry_feat], dim=-1)).squeeze(-1)

    def forward(self, obs_feat, obs_tgt, obs_aid, qry_feat, qry_aid):
        """TODO.

        Args:
            obs_feat: TODO.
            obs_tgt: TODO.
            obs_aid: TODO.
            qry_feat: TODO.
            qry_aid: TODO.

        Returns:
            TODO.
        """
        return self.decode(self.encode(obs_feat, obs_tgt, obs_aid), qry_feat, qry_aid)


class FittedCNP(SurfaceModel):
    """
    Trained MultiAssetCNP with normalisation statistics, ready for analytics.

    Targets are normalised as:  y_n = (log(y) - log_tgt_mean[asset]) / log_tgt_std[asset]
    so predict() always returns raw IV values (not log-IV).
    """

    def __init__(
        self,
        module: MultiAssetCNP,
        feat_mean: np.ndarray,     # (q_dim,)
        feat_std: np.ndarray,      # (q_dim,)
        log_tgt_mean: np.ndarray,  # (n_assets,)  per-asset mean of log(IV)
        log_tgt_std: np.ndarray,   # (n_assets,)  per-asset std  of log(IV)
    ):
        """TODO.

        Args:
            module: TODO.
            feat_mean: TODO.
            feat_std: TODO.
            log_tgt_mean: TODO.
            log_tgt_std: TODO.
        """
        self.module       = module
        self.feat_mean    = feat_mean
        self.feat_std     = feat_std
        self.log_tgt_mean = log_tgt_mean
        self.log_tgt_std  = log_tgt_std
        self._device      = next(module.parameters()).device

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _norm_feat(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feat_mean) / self.feat_std

    def _norm_tgt(self, y: np.ndarray, aids: np.ndarray) -> np.ndarray:
        """y: raw IV (...), aids: int (...) → normalised log-IV"""
        log_y = np.log(np.maximum(y, 1e-8))
        return (log_y - self.log_tgt_mean[aids]) / self.log_tgt_std[aids]

    def _denorm_tgt(self, y_n: np.ndarray, aids: np.ndarray) -> np.ndarray:
        """y_n: normalised (...), aids: int (...) → raw IV"""
        return np.exp(y_n * self.log_tgt_std[aids] + self.log_tgt_mean[aids])

    def _t(self, arr, dtype=torch.float32):
        return torch.from_numpy(np.asarray(arr, dtype=np.float32 if dtype == torch.float32
                                           else np.int64)).to(dtype=dtype, device=self._device)

    # ── Inference ─────────────────────────────────────────────────────────────

    def encode(self, obs_feats: np.ndarray, obs_aids: np.ndarray) -> np.ndarray:
        """TODO.

        Args:
            obs_feats: TODO.
            obs_aids: TODO.

        Returns:
            TODO.
        """
        B, n_ctx, _ = obs_feats.shape
        self.module.eval()
        with torch.no_grad():
            z = self.module.encode(
                self._t(self._norm_feat(obs_feats)),
                self._t(np.zeros((B, n_ctx), dtype=np.float32)),
                self._t(obs_aids, dtype=torch.long),
            )
        return z.cpu().numpy()

    def encode_with_targets(
        self,
        obs_feats: np.ndarray,
        obs_tgts: np.ndarray,
        obs_aids: np.ndarray,
    ) -> np.ndarray:
        """TODO.

        Args:
            obs_feats: TODO.
            obs_tgts: TODO.
            obs_aids: TODO.

        Returns:
            TODO.
        """
        self.module.eval()
        with torch.no_grad():
            z = self.module.encode(
                self._t(self._norm_feat(obs_feats)),
                self._t(self._norm_tgt(obs_tgts, obs_aids).astype(np.float32)),
                self._t(obs_aids, dtype=torch.long),
            )
        return z.cpu().numpy()

    def predict(
        self,
        obs_feats: np.ndarray,
        obs_aids: np.ndarray,
        obs_tgts: np.ndarray,
        qry_feats: np.ndarray,
        qry_aids: np.ndarray,
    ) -> np.ndarray:
        """TODO.

        Args:
            obs_feats: TODO.
            obs_aids: TODO.
            obs_tgts: TODO.
            qry_feats: TODO.
            qry_aids: TODO.

        Returns:
            TODO.
        """
        self.module.eval()
        with torch.no_grad():
            pred_n = self.module(
                self._t(self._norm_feat(obs_feats)),
                self._t(self._norm_tgt(obs_tgts, obs_aids).astype(np.float32)),
                self._t(obs_aids, dtype=torch.long),
                self._t(self._norm_feat(qry_feats)),
                self._t(qry_aids, dtype=torch.long),
            )
        return self._denorm_tgt(pred_n.cpu().numpy(), qry_aids)

    def decode_latent(
        self,
        z: np.ndarray,
        qry_feats: np.ndarray,
        qry_aids: np.ndarray,
    ) -> np.ndarray:
        """TODO.

        Args:
            z: TODO.
            qry_feats: TODO.
            qry_aids: TODO.

        Returns:
            TODO.
        """
        self.module.eval()
        with torch.no_grad():
            pred_n = self.module.decode(
                self._t(z),
                self._t(self._norm_feat(qry_feats)),
                self._t(qry_aids, dtype=torch.long),
            )
        return self._denorm_tgt(pred_n.cpu().numpy(), qry_aids)

    def encode_dataset(
        self,
        dataset,
        indices: np.ndarray,
        n_ctx: int,
        batch_size: int = 64,
    ) -> np.ndarray:
        """TODO.

        Args:
            dataset: TODO.
            indices: TODO.
            n_ctx: TODO.
            batch_size: TODO.

        Returns:
            TODO.
        """
        n_ctx = min(n_ctx, dataset.ctx_max)
        codes = []
        for b in range(0, len(indices), batch_size):
            idx = indices[b : b + batch_size]
            codes.append(self.encode_with_targets(
                dataset.query_feats[idx, :n_ctx],
                dataset.targets[idx, :n_ctx],
                dataset.asset_ids[idx, :n_ctx],
            ))
        return np.concatenate(codes, axis=0)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """TODO.

        Args:
            path: TODO.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":    self.module.state_dict(),
            "module_kwargs": _module_kwargs(self.module),
            "feat_mean":     self.feat_mean,
            "feat_std":      self.feat_std,
            "log_tgt_mean":  self.log_tgt_mean,
            "log_tgt_std":   self.log_tgt_std,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FittedCNP":
        """TODO.

        Args:
            path: TODO.
            device: TODO.

        Returns:
            TODO.
        """
        ck     = torch.load(path, map_location=device, weights_only=False)
        kwargs = dict(ck["module_kwargs"])
        kwargs.setdefault("dropout", 0.05)   # not saved in older checkpoints
        module = MultiAssetCNP(**kwargs).to(device)
        module.load_state_dict(ck["state_dict"])

        # Backward-compatible: old checkpoints stored a single (lm, T, iv) normalisation
        # vector under "norm_mean"/"norm_std" instead of the current per-asset log-IV scheme.
        if "feat_mean" in ck:
            return cls(module, ck["feat_mean"], ck["feat_std"],
                       ck["log_tgt_mean"], ck["log_tgt_std"])

        # Old format: norm_mean = [lm_mean, T_mean, iv_mean], norm_std same shape
        norm_mean = np.asarray(ck["norm_mean"], dtype=np.float32)
        norm_std  = np.asarray(ck["norm_std"],  dtype=np.float32)
        feat_mean = norm_mean[:module.obs_proj.in_features - module.asset_embed.embedding_dim - 1]
        feat_std  = norm_std [:module.obs_proj.in_features - module.asset_embed.embedding_dim - 1]
        n_assets  = module.n_assets
        # Reconstruct log-IV normalisation from global IV mean/std (index −1)
        iv_mean   = float(norm_mean[-1])
        iv_std    = float(norm_std[-1])
        # log_tgt_mean ≈ log(iv_mean) for all assets; log_tgt_std from delta method
        log_tgt_mean = np.full(n_assets, np.log(max(iv_mean, 1e-8)), dtype=np.float32)
        log_tgt_std  = np.full(n_assets, iv_std / max(iv_mean, 1e-8),  dtype=np.float32)
        return cls(module, feat_mean, feat_std, log_tgt_mean, log_tgt_std)


class FittedDeltaCNP(FittedCNP):
    """
    FittedCNP variant for the increment model.

    The model is trained to predict IV deltas (today - prior). The log_tgt_mean
    and log_tgt_std slots of the parent are repurposed to store delta_mean and
    delta_std.

    predict() accepts raw IV at context / query points plus the prior IV at
    those same points, and returns absolute IV (prior + predicted delta).
    """

    @property
    def delta_mean(self) -> np.ndarray:
        return self.log_tgt_mean

    @property
    def delta_std(self) -> np.ndarray:
        return self.log_tgt_std

    def _norm_tgt(self, y: np.ndarray, aids: np.ndarray) -> np.ndarray:
        """y: raw IV delta -> normalised delta."""
        return (y - self.delta_mean[aids]) / self.delta_std[aids]

    def _denorm_tgt(self, y_n: np.ndarray, aids: np.ndarray) -> np.ndarray:
        """y_n: normalised -> raw IV delta."""
        return y_n * self.delta_std[aids] + self.delta_mean[aids]

    def predict(
        self,
        obs_feats: np.ndarray,
        obs_aids: np.ndarray,
        obs_tgts: np.ndarray,
        qry_feats: np.ndarray,
        qry_aids: np.ndarray,
        obs_prior_iv: np.ndarray | None = None,
        qry_prior_iv: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        obs_tgts     : raw IV at context points
        obs_prior_iv : prior IV at context points (same shape as obs_tgts)
        qry_prior_iv : prior IV at all query points (same shape as output)
        Returns absolute IV (prior + predicted delta).
        """
        # Fall back to delta=0 for points where the prior couldn't be computed,
        # consistent with how trainer.py treats them (nan_to_num → 0).
        safe_prior = np.where(np.isfinite(obs_prior_iv), obs_prior_iv, obs_tgts)
        obs_delta  = obs_tgts - safe_prior
        self.module.eval()
        with torch.no_grad():
            pred_n = self.module(
                self._t(self._norm_feat(obs_feats)),
                self._t(self._norm_tgt(obs_delta, obs_aids).astype(np.float32)),
                self._t(obs_aids, dtype=torch.long),
                self._t(self._norm_feat(qry_feats)),
                self._t(qry_aids, dtype=torch.long),
            )
        pred_delta = self._denorm_tgt(pred_n.cpu().numpy(), qry_aids)
        return qry_prior_iv + pred_delta

    def encode_with_targets(
        self,
        obs_feats: np.ndarray,
        obs_tgts: np.ndarray,
        obs_aids: np.ndarray,
        obs_prior_iv: np.ndarray | None = None,
    ) -> np.ndarray:
        """obs_tgts: raw IV; obs_prior_iv: prior IV for computing the delta."""
        obs_delta = (obs_tgts - obs_prior_iv
                     if obs_prior_iv is not None
                     else np.zeros_like(obs_tgts))
        self.module.eval()
        with torch.no_grad():
            z = self.module.encode(
                self._t(self._norm_feat(obs_feats)),
                self._t(self._norm_tgt(obs_delta, obs_aids).astype(np.float32)),
                self._t(obs_aids, dtype=torch.long),
            )
        return z.cpu().numpy()

    def encode_dataset(
        self,
        dataset,
        indices: np.ndarray,
        n_ctx: int,
        batch_size: int = 64,
    ) -> np.ndarray:
        n_ctx  = min(n_ctx, dataset.ctx_max)
        codes  = []
        for b in range(0, len(indices), batch_size):
            idx        = indices[b : b + batch_size]
            prior_ctx  = (dataset.prior_targets[idx, :n_ctx]
                          if dataset.prior_targets is not None else None)
            codes.append(self.encode_with_targets(
                dataset.query_feats[idx, :n_ctx],
                dataset.targets[idx, :n_ctx],
                dataset.asset_ids[idx, :n_ctx],
                obs_prior_iv=prior_ctx,
            ))
        return np.concatenate(codes, axis=0)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":    self.module.state_dict(),
            "module_kwargs": _module_kwargs(self.module),
            "feat_mean":     self.feat_mean,
            "feat_std":      self.feat_std,
            "delta_mean":    self.delta_mean,
            "delta_std":     self.delta_std,
            "delta_mode":    True,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FittedDeltaCNP":
        ck     = torch.load(path, map_location=device, weights_only=False)
        kwargs = dict(ck["module_kwargs"])
        kwargs.setdefault("dropout", 0.05)
        module = MultiAssetCNP(**kwargs).to(device)
        module.load_state_dict(ck["state_dict"])
        return cls(module, ck["feat_mean"], ck["feat_std"],
                   ck["delta_mean"], ck["delta_std"])


def transfer_weights(src: MultiAssetCNP, dst: MultiAssetCNP) -> int:
    """
    Copy all weights from src to dst, skipping asset_embed and prior which are
    n_assets-dependent. All shared transformer and decoder layers transfer directly
    because self-attention weight shapes are independent of sequence length.
    Returns the count of parameter tensors actually copied.
    """
    skip   = {"asset_embed.weight", "prior"}
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    n_copied = 0
    for k in list(dst_sd):
        if k not in skip and k in src_sd and src_sd[k].shape == dst_sd[k].shape:
            dst_sd[k] = src_sd[k].clone()
            n_copied += 1
    dst.load_state_dict(dst_sd)
    return n_copied


def _module_kwargs(m: MultiAssetCNP) -> dict:
    return dict(
        n_assets      = m.n_assets,
        q_dim         = m.obs_proj.in_features - m.asset_embed.embedding_dim - 1,
        d_asset       = m.asset_embed.embedding_dim,
        d_model       = m.obs_norm.normalized_shape[0],
        n_heads_obs   = m.obs_enc.layers[0].self_attn.num_heads,
        n_heads_cross = m.cross_enc.layers[0].self_attn.num_heads,
        n_layers_obs  = len(m.obs_enc.layers),
        n_layers_cross= len(m.cross_enc.layers),
        d_latent      = m.d_latent,
        d_hidden      = m.decoder[0].out_features,
        n_hidden_dec  = sum(1 for l in m.decoder if isinstance(l, nn.Linear)) - 1,
    )
