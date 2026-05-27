from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = str(
    REPO_ROOT / "data" / "scripts" / "bulk_download" / "output" / "group_tech_us.csv"
)


@dataclass
class DataConfig:
    csv_path:     str = CSV_PATH
    n_train_days: int | None = None   # None = use all training dates
    n_val_days:   int | None = None
    val_frac:     float = 0.15


@dataclass
class ModelConfig:
    # ~126k params — good fit for ~1000 training days, ~30–45 min to train
    d_asset:        int   = 8    # embedding dim per asset (also = n_assets here)
    d_model:        int   = 32
    n_heads_obs:    int   = 4
    n_layers_obs:   int   = 4
    n_heads_cross:  int   = 4
    n_layers_cross: int   = 3
    d_latent:       int   = 16
    d_hidden:       int   = 128
    n_hidden_dec:   int   = 3
    dropout:        float = 0.05


@dataclass
class TrainConfig:
    n_epochs:   int   = 100
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


@dataclass
class Config:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    train:     TrainConfig     = field(default_factory=TrainConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    device:    str             = "cuda"
    seed:      int             = 0
    out_dir:   str             = str(REPO_ROOT / "results" / "vol_surface_grouptech")
