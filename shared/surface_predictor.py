"""
Common prediction interface shared between dgraph and neural_processes.

FittedCNP already satisfies SurfacePredictor.
DGraphSurfacePredictor (dgraph/examples/vol_surface/source/predictor.py) wraps a
fitted Graph for parametric models.
"""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class SurfacePredictor(Protocol):
    """
    Predict implied vol at query coordinates given context observations.

    Parametric implementations (dgraph) may ignore obs_* entirely because the
    model is already fitted; neural implementations (CNP) use them for
    in-context conditioning.
    """

    def predict(
        self,
        obs_feats: np.ndarray,  # (B, N_obs, Q_dim)
        obs_aids:  np.ndarray,  # (B, N_obs)  int, 0-based asset index
        obs_tgts:  np.ndarray,  # (B, N_obs)  observed IV values
        qry_feats: np.ndarray,  # (B, N_qry, Q_dim)
        qry_aids:  np.ndarray,  # (B, N_qry)  int
    ) -> np.ndarray:            # (B, N_qry)  predicted IV
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
