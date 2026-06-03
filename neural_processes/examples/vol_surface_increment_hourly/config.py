from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = str(
    REPO_ROOT / "data" / "scripts" / "bulk_download" / "output"
    / "group_tech_hourly" / "group_tech_us_hourly.csv"
)


@dataclass
class DataConfig:
    csv_path:    str = CSV_PATH
    obs_col:     str = "datetime"
    n_eval_days: int = 200          # last 200 hourly snapshots (~25 trading days)


@dataclass
class ModelConfig:
    d_asset:        int   = 8
    d_model:        int   = 64
    n_heads_obs:    int   = 4
    n_layers_obs:   int   = 4
    n_heads_cross:  int   = 4
    n_layers_cross: int   = 4
    d_latent:       int   = 32
    d_hidden:       int   = 192
    n_hidden_dec:   int   = 3
    dropout:        float = 0.1


@dataclass
class TrainConfig:
    n_epochs:   int   = 80
    batch_size: int   = 32
    lr:         float = 3e-4
    ctx_min:    int   = 10
    log_every:  int   = 10


@dataclass
class AnalyticsConfig:
    ctx_sizes_recon:      list = field(default_factory=lambda: [5, 20, 50, 100])
    ctx_sizes_rmse_curve: list = field(default_factory=lambda: [5, 20, 50, 100, 200])
    n_ctx_per_maturity:   int  = 100
    n_ctx_latent:         int  = 200
    n_ctx_zeroshot:       int  = 50


@dataclass
class Config:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    train:     TrainConfig     = field(default_factory=TrainConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    device:    str             = "cuda"
    seed:      int             = 0
    out_dir:   str             = str(REPO_ROOT / "results" / "vol_surface_increment_hourly")
