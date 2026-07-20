"""
CNP-specific latent-space analytics (only meaningful for CNPModel).

Encodes each (day, asset) context into the CNP's per-asset latent code, projects to 2-D
PCA, and (for synthetic data with ground-truth params) colours the scatter by a chosen
DGP parameter — showing whether the learned latent organises by the generating factors.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def encode_latents(cnp_model, dataset, indices, n_ctx: int = 100):
    """Return (Z, asset_ids) where Z is (len(indices)*n_assets, d_latent)."""
    fitted = cnp_model.fitted
    Zs, aids = [], []
    for t in indices:
        m = dataset.valid_mask(t)
        cpool = np.where(m)[0][:dataset.ctx_max]
        feats = dataset.query_feats[t, cpool][None, :n_ctx]
        tgts = dataset.targets[t, cpool][None, :n_ctx]
        a = dataset.asset_ids[t, cpool][None, :n_ctx]
        z = fitted.encode_with_targets(feats, tgts, a)[0]   # (n_assets, d_latent)
        Zs.append(z); aids.append(np.arange(z.shape[0]))
    return np.concatenate(Zs), np.concatenate(aids)


def plot_latent_pca(cnp_model, dataset, indices=None, n_ctx: int = 100,
                    param_idx: int | None = None, param_name: str = "param",
                    out_path: str | None = None):
    """2-D PCA of CNP latents, optionally coloured by a ground-truth DGP parameter."""
    if indices is None:
        indices = dataset.val_idx()
    Z, aids = encode_latents(cnp_model, dataset, indices, n_ctx)
    Zc = Z - Z.mean(0)
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    z2 = Zc @ Vt[:2].T
    evr = (s[:2] ** 2) / (s ** 2).sum()

    color = None
    if param_idx is not None and dataset.params is not None:
        color = np.concatenate([dataset.params[t, :, param_idx] for t in indices])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=color, cmap="viridis", s=10, alpha=0.7)
    if color is not None:
        fig.colorbar(sc, label=param_name)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
    ax.set_title(f"{cnp_model.name} latent PCA"
                 + (f" — coloured by {param_name}" if color is not None else ""))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig
