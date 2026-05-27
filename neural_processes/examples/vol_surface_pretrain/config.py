from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = str(
    REPO_ROOT / "data" / "scripts" / "bulk_download" / "output" / "group_tech_us.csv"
)


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class HestonDataConfig:
    train_path:   str       = str(REPO_ROOT / "data" / "synthetic" / "heston_multiasset_training.npz")
    ood_path:     str | None = None
    n_train_days: int | None = None
    n_val_days:   int | None = None


@dataclass
class GroupTechDataConfig:
    csv_path:     str       = CSV_PATH
    n_train_days: int | None = None
    n_val_days:   int | None = None
    val_frac:     float     = 0.15


# ── Architecture (shared between pre-train and fine-tune phases) ──────────────

@dataclass
class ModelConfig:
    # ~126k params — identical to vol_surface_grouptech so transfer is a fair comparison.
    # Heston pre-train uses this architecture with 5 assets; fine-tune re-uses all
    # weights except asset_embed and prior (which are reset for 8 GroupTech assets).
    d_asset:        int   = 8
    d_model:        int   = 32
    n_heads_obs:    int   = 4
    n_layers_obs:   int   = 4
    n_heads_cross:  int   = 4
    n_layers_cross: int   = 3
    d_latent:       int   = 16
    d_hidden:       int   = 128
    n_hidden_dec:   int   = 3
    dropout:        float = 0.05


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class PretrainTrainConfig:
    """Heston pre-training — 75 epochs is plenty given 5000 days of synthetic data."""
    n_epochs:   int   = 75
    batch_size: int   = 32
    lr:         float = 3e-4
    ctx_min:    int   = 3
    log_every:  int   = 10


@dataclass
class FineTuneTrainConfig:
    """Group Tech fine-tune — lower LR preserves the pre-trained transformer weights."""
    n_epochs:   int   = 75
    batch_size: int   = 32
    lr:         float = 5e-5
    ctx_min:    int   = 3
    log_every:  int   = 10


@dataclass
class AnalyticsConfig:
    ctx_sizes_recon:      list = field(default_factory=lambda: [5, 20, 50, 100])
    ctx_sizes_rmse_curve: list = field(default_factory=lambda: [5, 20, 50, 100, 200])
    n_ctx_per_maturity:   int  = 100
    n_ctx_zeroshot:       int  = 50


# ── Top-level config ──────────────────────────────────────────────────────────

@dataclass
class Config:
    pretrain_data: HestonDataConfig      = field(default_factory=HestonDataConfig)
    pretrain_train: PretrainTrainConfig  = field(default_factory=PretrainTrainConfig)
    data:          GroupTechDataConfig   = field(default_factory=GroupTechDataConfig)
    model:         ModelConfig           = field(default_factory=ModelConfig)
    train:         FineTuneTrainConfig   = field(default_factory=FineTuneTrainConfig)
    analytics:     AnalyticsConfig       = field(default_factory=AnalyticsConfig)
    device:        str                   = "cuda"
    seed:          int                   = 0
    out_dir:       str                   = str(REPO_ROOT / "results" / "vol_surface_pretrain")
