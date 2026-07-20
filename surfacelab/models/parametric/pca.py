"""
Functional-PCA surface model.

Unlike the per-day fitters, PCA has a real `train()`: it learns a joint multi-asset
factor basis (`FactorBasis`) from the training surfaces.  `predict` then fits factor
coefficients to today's context by ridge least squares and reconstructs the surface
at the query points.  Cross-asset structure is captured by the joint basis.
"""
from __future__ import annotations

import numpy as np

from surfacelab.core.model import SurfaceModel
from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction
from surfacelab.models.factors import FactorBasis, grid_from_dataset


class PCAModel(SurfaceModel):
    name = "pca"

    def __init__(self, n_components: int = 20, ridge: float = 1e-4):
        self.n_components = n_components
        self.ridge = ridge
        self.basis: FactorBasis | None = None

    def train(self, data, *, saved: bool = False, force: bool = False) -> None:
        self.data_tag = data.meta.get("dgp", self.data_tag)
        grid = grid_from_dataset(data)
        X = grid.stack_days(data, data.train_idx())      # (n_train, n_total)
        self.basis = FactorBasis(self.n_components, grid).fit(X)

    def _fit_factors(self, context: Quotes) -> np.ndarray:
        H = self.basis.observation_matrix(context.feats, context.asset_id)   # (Nc, k)
        y = context.iv - self.basis.mean_at(context.feats, context.asset_id)
        # Ridge-regularised normal equations: (H'H + λI) z = H'y
        k = H.shape[1]
        A = H.T @ H + self.ridge * np.eye(k)
        b = H.T @ y
        return np.linalg.solve(A, b)

    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self.basis is None:
            raise RuntimeError("PCAModel.predict called before train()")
        z = self._fit_factors(context)
        Hq = self.basis.observation_matrix(query.feats, query.asset_id)
        iv = Hq @ z + self.basis.mean_at(query.feats, query.asset_id)
        return SurfacePrediction(iv=np.maximum(iv, 1e-8))
