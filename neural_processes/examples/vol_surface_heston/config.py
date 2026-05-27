from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class DataConfig:
    train_path:   str = str(REPO_ROOT / "data" / "synthetic" / "heston_multiasset_training.npz")
    ood_path:     str = str(REPO_ROOT / "data" / "synthetic" / "heston_multiasset_ood_test.npz")
    n_train_days: int = 500
    n_val_days:   int = 100


@dataclass
class ModelConfig:
    d_asset:       int   = 8
    d_model:       int   = 32
    n_heads_obs:   int   = 4
    n_layers_obs:  int   = 2
    n_heads_cross: int   = 4
    n_layers_cross:int   = 3
    d_latent:      int   = 16
    d_hidden:      int   = 128
    n_hidden_dec:  int   = 3
    dropout:       float = 0.05


@dataclass
class TrainConfig:
    n_epochs:   int   = 150
    batch_size: int   = 32
    lr:         float = 3e-4
    ctx_min:    int   = 3
    log_every:  int   = 10


@dataclass
class AnalyticsConfig:
    ctx_sizes_recon:      list = field(default_factory=lambda: [5, 20, 50, 100])
    ctx_sizes_rmse_curve: list = field(default_factory=lambda: [5, 20, 50, 100, 200])
    n_ctx_per_maturity:   int  = 100
    n_ctx_zeroshot:       int  = 50
    n_ctx_latent:         int  = 400
    # Heston params layout: [v0, kappa, theta, xi, rho]
    latent_param_names:   list = field(default_factory=lambda: ["θ", "κ", "ξ", "ρ"])
    latent_param_indices: list = field(default_factory=lambda: [2, 1, 3, 4])
    latent_cmaps:         list = field(default_factory=lambda: ["RdYlGn_r", "PuBu", "YlOrRd", "coolwarm"])


@dataclass
class Config:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    train:     TrainConfig     = field(default_factory=TrainConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    device:    str             = "cuda"
    seed:      int             = 0
    out_dir:   str             = str(REPO_ROOT / "results" / "vol_surface_neural")
