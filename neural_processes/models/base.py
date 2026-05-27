from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class SurfaceModel(ABC):
    """
    Inference interface every fitted surface model must satisfy.
    Training is handled by Trainer; this interface is purely for prediction and analytics.
    """

    @abstractmethod
    def encode(
        self,
        obs_feats: np.ndarray,   # (B, n_ctx, Q_dim)
        obs_aids: np.ndarray,    # (B, n_ctx)  int
    ) -> np.ndarray:             # (B, N_assets, D_latent)
        ...

    @abstractmethod
    def predict(
        self,
        obs_feats: np.ndarray,   # (B, n_ctx, Q_dim)
        obs_aids: np.ndarray,    # (B, n_ctx)  int
        obs_tgts: np.ndarray,    # (B, n_ctx)  float — target values at context points
        qry_feats: np.ndarray,   # (B, Q, Q_dim)
        qry_aids: np.ndarray,    # (B, Q)  int
    ) -> np.ndarray:             # (B, Q)  — denormalised
        ...

    @abstractmethod
    def decode_latent(
        self,
        z: np.ndarray,           # (B, N_assets, D_latent)
        qry_feats: np.ndarray,   # (B, Q, Q_dim)
        qry_aids: np.ndarray,    # (B, Q)  int
    ) -> np.ndarray:             # (B, Q)  — denormalised
        ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "SurfaceModel": ...
