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
        """TODO.

        Args:
            obs_feats: TODO.
            obs_aids: TODO.

        Returns:
            TODO.
        """
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
        ...

    @abstractmethod
    def decode_latent(
        self,
        z: np.ndarray,           # (B, N_assets, D_latent)
        qry_feats: np.ndarray,   # (B, Q, Q_dim)
        qry_aids: np.ndarray,    # (B, Q)  int
    ) -> np.ndarray:             # (B, Q)  — denormalised
        """TODO.

        Args:
            z: TODO.
            qry_feats: TODO.
            qry_aids: TODO.

        Returns:
            TODO.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """TODO.

        Args:
            path: TODO.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "SurfaceModel":
        """TODO.

        Args:
            path: TODO.

        Returns:
            TODO.
        """
        ...
