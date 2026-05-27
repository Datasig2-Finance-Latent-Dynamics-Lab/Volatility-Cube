from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from neural_processes.data.base import SurfaceDataset


def encode_dataset(model, dataset, indices, n_ctx):
    return model.encode_dataset(dataset, indices, n_ctx)


def pca_latent(z):
    """z: (N,A,D) -> z_2d (N*A,2), var_exp (D,)"""
    Z_flat = z.reshape(-1, z.shape[-1])
    Z_c    = Z_flat - Z_flat.mean(0)
    _, S, Vt = np.linalg.svd(Z_c, full_matrices=False)
    return Z_c @ Vt[:2].T, S ** 2 / (S ** 2).sum()


def plot_pca_colored(z_2d, var_exp, params, param_names, param_indices,
                     cmaps=None, out_path=None):
    cmaps = cmaps or ["RdYlGn_r", "PuBu", "YlOrRd", "coolwarm"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for idx, (pname, pidx, cmap_name) in enumerate(zip(param_names, param_indices, cmaps)):
        if idx >= 4:
            break
        ax = axes.ravel()[idx]
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=params[:, :, pidx].ravel(),
                        cmap=cmap_name, s=12, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label=pname, shrink=0.85)
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=9)
        ax.set_title(f"Latent space — {pname}", fontsize=10)
    fig.suptitle("Latent Space PCA", fontsize=12)
    _maybe_save(fig, out_path)
    return fig


def plot_r2_heatmap(z_2d, params, param_names, param_indices,
                    asset_names=None, out_path=None):
    from numpy.linalg import lstsq
    N_days, n_assets, _ = params.shape
    names    = asset_names or [f"asset_{i}" for i in range(n_assets)]
    n_params = len(param_names)
    R2_pc    = np.zeros((2, n_params, n_assets))

    for p_idx, pidx in enumerate(param_indices):
        for a in range(n_assets):
            rows  = np.arange(N_days) * n_assets + a
            pvals = params[:, a, pidx]
            X     = np.column_stack([np.ones(N_days), pvals])
            for pc in range(2):
                y = z_2d[rows, pc]
                y_hat = X @ lstsq(X, y, rcond=None)[0]
                ss_tot = np.sum((y - y.mean()) ** 2)
                R2_pc[pc, p_idx, a] = max(1 - np.sum((y - y_hat) ** 2) / (ss_tot + 1e-12), 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for pc, ax in zip(range(2), axes):
        im = ax.imshow(R2_pc[pc], aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n_assets)); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticks(range(n_params)); ax.set_yticklabels(param_names)
        ax.set_title(f"R²: PC{pc+1} ~ parameter (per asset)")
        plt.colorbar(im, ax=ax, label="R²", shrink=0.85)
        for i in range(n_params):
            for j in range(n_assets):
                v = R2_pc[pc, i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.7 else "black")
    _maybe_save(fig, out_path)
    return fig


def plot_latent_interpolation(model, dataset, day_lo, day_hi, move_asset=0,
                               n_interp=7, vis_lm=None, vis_mats=None, out_path=None):
    if vis_lm   is None: vis_lm   = np.linspace(-0.40, 0.40, 20).astype(np.float32)
    if vis_mats is None: vis_mats = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], np.float32)

    n_assets    = dataset.n_assets
    asset_names = dataset.meta.get("asset_names", [f"asset_{i}" for i in range(n_assets)])

    z_lo = model.encode_dataset(dataset, np.array([day_lo]), n_ctx=dataset.ctx_max)[0]
    z_hi = model.encode_dataset(dataset, np.array([day_hi]), n_ctx=dataset.ctx_max)[0]

    grid_lm = np.repeat(vis_lm, len(vis_mats))
    grid_T  = np.tile(vis_mats, len(vis_lm))
    extra   = np.zeros((len(grid_lm), dataset.q_dim - 2), dtype=np.float32)
    vis_grid = np.concatenate([grid_lm[:, None], grid_T[:, None], extra], axis=1)[np.newaxis]

    alphas = np.linspace(0, 1, n_interp)
    cmap   = plt.cm.viridis
    cols   = [cmap(i / max(len(vis_mats) - 1, 1)) for i in range(len(vis_mats))]

    fig, axes = plt.subplots(n_assets, n_interp,
                              figsize=(3 * n_interp, 2.5 * n_assets), sharey="row")
    if n_assets == 1:
        axes = axes[np.newaxis, :]

    for j, alpha in enumerate(alphas):
        z_interp = z_lo.copy()[np.newaxis]
        z_interp[0, move_asset] = (1 - alpha) * z_lo[move_asset] + alpha * z_hi[move_asset]
        for a in range(n_assets):
            ax = axes[a, j]
            qry_aids = np.full((1, len(grid_lm)), a, dtype=np.int64)
            pred_iv  = model.decode_latent(z_interp, vis_grid, qry_aids)[0]
            for t_idx, T in enumerate(vis_mats):
                m = grid_T == T
                ax.plot(vis_lm, pred_iv[m], color=cols[t_idx], lw=1.2)
            if a == 0:
                ax.set_title(f"α={alpha:.2f}", fontsize=8)
            if j == 0:
                suffix = " ← moved" if a == move_asset else ""
                ax.set_ylabel(f"{asset_names[a]}{suffix}", fontsize=8)
            ax.set_xlabel("lm", fontsize=6)
            ax.tick_params(labelsize=5)

    fig.suptitle(f"Latent interpolation: only {asset_names[move_asset]} moves", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vis_mats[0], vis_mats[-1]))
    fig.colorbar(sm, ax=axes, label="Maturity", fraction=0.01)
    _maybe_save(fig, out_path)
    return fig


def _maybe_save(fig, path):
    if path:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.tight_layout()
