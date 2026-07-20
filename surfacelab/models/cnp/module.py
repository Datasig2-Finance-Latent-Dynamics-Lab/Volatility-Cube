"""
MultiAssetCNP network + Fitted(Delta)CNP inference wrappers.

Attentive (Transformer) Neural Process.  There is NO latent bottleneck and NO mean
pooling: every query point attends directly to the encoded context quotes, so a
prediction genuinely depends on the context set (more/better quotes → better fit).

  Context encoder : token_i = [asset_embed(a_i), feats_i, target_i] → d_model, then
                    full self-attention across ALL context points of ALL assets.  Cross-
                    asset structure is learned through attention + the asset embedding —
                    no separate cross-asset stage, no per-asset pooling.
  Decoder         : query token = [asset_embed(a_q), feats_q] → d_model, then stacked
                    cross-attention blocks (query attends to the encoded context) → MLP head.

Padding/absent context points are masked out of every attention (key_padding_mask).
The surfacelab contract is satisfied by the `CNPModel` adapter in `model.py`.
"""
from __future__ import annotations
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_IV_FLOOR = 1e-8           # IV positivity floor before the log-target transform
_DEFAULT_DROPOUT = 0.1     # default dropout (and fallback for checkpoints lacking the key)


def _key_padding_mask(obs_valid: torch.Tensor | None) -> torch.Tensor | None:
    """Turn a (B, C) bool *validity* mask into a MultiheadAttention key_padding_mask
    (True == ignore).  Rows with no valid key are un-masked entirely so attention stays
    finite (their query outputs are discarded downstream anyway)."""
    if obs_valid is None:
        return None
    kpm = ~obs_valid.bool()                 # True where padding/absent
    all_pad = kpm.all(dim=1)
    if all_pad.any():
        kpm = kpm.clone()
        kpm[all_pad] = False
    return kpm


class _FourierFeatures(nn.Module):
    """Random Fourier features on the (k, T) coordinates (Tancik et al. 2020).

    Maps each coordinate vector x → [x, sin(2π·xB), cos(2π·xB)] with a fixed random
    Gaussian B.  Coordinate-based attention needs high-frequency position signal to match
    a query to *nearby* context quotes sharply — a plain linear projection of (k, T) is too
    smooth, so even with the whole surface as context the model can't reproduce it.  B is a
    saved buffer, so a loaded checkpoint sees the same frequencies it trained with.
    """

    def __init__(self, in_dim, n_freqs, scale=2.0):
        super().__init__()
        self.out_dim = in_dim + 2 * n_freqs
        self.register_buffer("B", torch.randn(in_dim, n_freqs) * scale)

    def forward(self, x):
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN: query tokens attend to a fixed context memory."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, q, mem, key_padding_mask=None):
        h = self.norm_q(q)
        a, _ = self.attn(h, mem, mem, key_padding_mask=key_padding_mask,
                         need_weights=False)
        q = q + a
        q = q + self.ff(self.norm_ff(q))
        return q


class MultiAssetCNP(nn.Module):
    """Attentive Neural Process for multi-asset vol surfaces (see module docstring)."""

    def __init__(self, n_assets, q_dim, d_asset=16, d_model=64, n_heads=4,
                 n_layers_enc=3, n_layers_dec=3, d_hidden=128,
                 n_fourier=16, fourier_scale=2.0, dropout=_DEFAULT_DROPOUT):
        super().__init__()
        self.n_assets = n_assets
        self.q_dim = q_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.n_fourier = n_fourier
        self.fourier_scale = fourier_scale

        self.asset_embed = nn.Embedding(n_assets, d_asset)
        self.feat_map = (_FourierFeatures(q_dim, n_fourier, fourier_scale)
                         if n_fourier > 0 else None)
        f_dim = self.feat_map.out_dim if self.feat_map is not None else q_dim
        self.obs_proj = nn.Linear(d_asset + f_dim + 1, d_model)   # +1 for the target
        self.qry_proj = nn.Linear(d_asset + f_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.ctx_enc = nn.TransformerEncoder(enc_layer, n_layers_enc)
        self.ctx_norm = nn.LayerNorm(d_model)

        self.dec_blocks = nn.ModuleList(
            [_CrossAttnBlock(d_model, n_heads, d_model * 4, dropout)
             for _ in range(n_layers_dec)])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hidden, 1))

    def _feats(self, x):
        return self.feat_map(x) if self.feat_map is not None else x

    # ── encoder: context quotes → per-point memory tokens ──────────────────────
    def encode_memory(self, obs_feat, obs_tgt, obs_aid, obs_valid=None):
        e = self.asset_embed(obs_aid)
        x = torch.cat([e, self._feats(obs_feat), obs_tgt.unsqueeze(-1)], dim=-1)
        kpm = _key_padding_mask(obs_valid)
        h = self.ctx_enc(self.obs_proj(x), src_key_padding_mask=kpm)
        return self.ctx_norm(h), kpm                              # (B, C, d_model), mask

    # ── decoder: query points cross-attend to the context memory ───────────────
    def decode(self, mem, kpm, qry_feat, qry_aid):
        e = self.asset_embed(qry_aid)
        q = self.qry_proj(torch.cat([e, self._feats(qry_feat)], dim=-1))
        for blk in self.dec_blocks:
            q = blk(q, mem, key_padding_mask=kpm)
        return self.head(q).squeeze(-1)

    def forward(self, obs_feat, obs_tgt, obs_aid, qry_feat, qry_aid, obs_valid=None):
        mem, kpm = self.encode_memory(obs_feat, obs_tgt, obs_aid, obs_valid)
        return self.decode(mem, kpm, qry_feat, qry_aid)

    # ── analytics hook: per-asset pooled context encoding (replaces the old latent)
    def encode(self, obs_feat, obs_tgt, obs_aid, obs_valid=None):
        """Per-asset mean of the encoded context tokens — a summary representation for
        analytics (latent.py).  Not used by predict(); predictions use the full memory."""
        mem, _ = self.encode_memory(obs_feat, obs_tgt, obs_aid, obs_valid)
        B = mem.shape[0]
        valid = (torch.ones(obs_aid.shape, device=mem.device, dtype=torch.bool)
                 if obs_valid is None else obs_valid.bool())
        one_hot = (F.one_hot(obs_aid, self.n_assets).float()
                   * valid.unsqueeze(-1).float())
        sums = torch.einsum("bcp,bcd->bpd", one_hot, mem)
        counts = one_hot.sum(1).unsqueeze(-1).clamp(min=1)
        return sums / counts                                      # (B, n_assets, d_model)


class FittedCNP:
    """Trained MultiAssetCNP + normalisation stats; numpy predict() API.

    Targets are normalised as y_n = (log(y) - log_tgt_mean[asset]) / log_tgt_std[asset],
    so predict() always returns raw IV.
    """

    def __init__(self, module, feat_mean, feat_std, log_tgt_mean, log_tgt_std):
        self.module = module
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.log_tgt_mean = log_tgt_mean
        self.log_tgt_std = log_tgt_std
        self._device = next(module.parameters()).device

    def _norm_feat(self, x):
        return (x - self.feat_mean) / self.feat_std

    def _norm_tgt(self, y, aids):
        log_y = np.log(np.maximum(y, _IV_FLOOR))
        return (log_y - self.log_tgt_mean[aids]) / self.log_tgt_std[aids]

    def _denorm_tgt(self, y_n, aids):
        return np.exp(y_n * self.log_tgt_std[aids] + self.log_tgt_mean[aids])

    def _t(self, arr, dtype=torch.float32):
        np_dtype = np.float32 if dtype == torch.float32 else np.int64
        return torch.from_numpy(np.asarray(arr, dtype=np_dtype)).to(dtype=dtype,
                                                                    device=self._device)

    def encode(self, obs_feats, obs_aids):
        B, n_ctx, _ = obs_feats.shape
        self.module.eval()
        with torch.no_grad():
            z = self.module.encode(
                self._t(self._norm_feat(obs_feats)),
                self._t(np.zeros((B, n_ctx), dtype=np.float32)),
                self._t(obs_aids, dtype=torch.long))
        return z.cpu().numpy()

    def encode_with_targets(self, obs_feats, obs_tgts, obs_aids):
        self.module.eval()
        with torch.no_grad():
            z = self.module.encode(
                self._t(self._norm_feat(obs_feats)),
                self._t(self._norm_tgt(obs_tgts, obs_aids).astype(np.float32)),
                self._t(obs_aids, dtype=torch.long))
        return z.cpu().numpy()

    def predict(self, obs_feats, obs_aids, obs_tgts, qry_feats, qry_aids):
        self.module.eval()
        with torch.no_grad():
            pred_n = self.module(
                self._t(self._norm_feat(obs_feats)),
                self._t(self._norm_tgt(obs_tgts, obs_aids).astype(np.float32)),
                self._t(obs_aids, dtype=torch.long),
                self._t(self._norm_feat(qry_feats)),
                self._t(qry_aids, dtype=torch.long))
        return self._denorm_tgt(pred_n.cpu().numpy(), qry_aids)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.module.state_dict(),
            "module_kwargs": _module_kwargs(self.module),
            "feat_mean": self.feat_mean, "feat_std": self.feat_std,
            "log_tgt_mean": self.log_tgt_mean, "log_tgt_std": self.log_tgt_std,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        # filter kwargs to the current constructor signature so checkpoints from earlier
        # architectures load what they can (and fail loudly only on a genuine shape clash).
        sig = set(inspect.signature(MultiAssetCNP.__init__).parameters)
        kwargs = {k: v for k, v in dict(ck["module_kwargs"]).items() if k in sig}
        kwargs.setdefault("dropout", _DEFAULT_DROPOUT)
        module = MultiAssetCNP(**kwargs).to(device)
        module.load_state_dict(ck["state_dict"])
        # Target-normalisation stats: current key is {log_tgt,delta}_mean/std; older
        # checkpoints stored them under a generic tgt_mean/tgt_std (+ a `delta` flag).
        is_delta = ("delta_mean" in ck) or bool(ck.get("delta", False)) \
            or bool(ck.get("delta_mode", False))
        tgt_mean = ck.get("delta_mean", ck.get("log_tgt_mean", ck.get("tgt_mean")))
        tgt_std = ck.get("delta_std", ck.get("log_tgt_std", ck.get("tgt_std")))
        target_cls = FittedDeltaCNP if is_delta else cls
        return target_cls(module, ck["feat_mean"], ck["feat_std"], tgt_mean, tgt_std)


class FittedDeltaCNP(FittedCNP):
    """Increment CNP: predicts IV deltas (today - prior). log_tgt_* slots hold delta_*."""

    @property
    def delta_mean(self):
        return self.log_tgt_mean

    @property
    def delta_std(self):
        return self.log_tgt_std

    def _norm_tgt(self, y, aids):
        return (y - self.delta_mean[aids]) / self.delta_std[aids]

    def _denorm_tgt(self, y_n, aids):
        return y_n * self.delta_std[aids] + self.delta_mean[aids]

    def predict(self, obs_feats, obs_aids, obs_tgts, qry_feats, qry_aids,
                obs_prior_iv=None, qry_prior_iv=None):
        safe_prior = np.where(np.isfinite(obs_prior_iv), obs_prior_iv, obs_tgts)
        obs_delta = obs_tgts - safe_prior
        self.module.eval()
        with torch.no_grad():
            pred_n = self.module(
                self._t(self._norm_feat(obs_feats)),
                self._t(self._norm_tgt(obs_delta, obs_aids).astype(np.float32)),
                self._t(obs_aids, dtype=torch.long),
                self._t(self._norm_feat(qry_feats)),
                self._t(qry_aids, dtype=torch.long))
        pred_delta = self._denorm_tgt(pred_n.cpu().numpy(), qry_aids)
        return qry_prior_iv + pred_delta

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.module.state_dict(),
            "module_kwargs": _module_kwargs(self.module),
            "feat_mean": self.feat_mean, "feat_std": self.feat_std,
            "delta_mean": self.delta_mean, "delta_std": self.delta_std,
            "delta_mode": True,
        }, path)

    # load() is inherited from FittedCNP, which detects the delta flag and returns a
    # FittedDeltaCNP automatically.


def transfer_weights(src, dst) -> int:
    """Copy shared weights src→dst, skipping asset_embed (n_assets-dependent)."""
    skip = {"asset_embed.weight"}
    src_sd, dst_sd = src.state_dict(), dst.state_dict()
    n = 0
    for k in list(dst_sd):
        if k not in skip and k in src_sd and src_sd[k].shape == dst_sd[k].shape:
            dst_sd[k] = src_sd[k].clone(); n += 1
    dst.load_state_dict(dst_sd)
    return n


def _module_kwargs(m: MultiAssetCNP) -> dict:
    return dict(
        n_assets=m.n_assets,
        q_dim=m.q_dim,
        d_asset=m.asset_embed.embedding_dim,
        d_model=m.d_model,
        n_heads=m.n_heads,
        n_layers_enc=len(m.ctx_enc.layers),
        n_layers_dec=len(m.dec_blocks),
        d_hidden=m.d_hidden,
        n_fourier=m.n_fourier,
        fourier_scale=m.fourier_scale,
    )
