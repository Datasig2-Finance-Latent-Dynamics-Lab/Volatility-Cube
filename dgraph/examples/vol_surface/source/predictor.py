"""
Wraps a fitted vol-surface Graph as a SurfacePredictor.

This lets dgraph vol-surface models plug into the shared analytics in
shared/analytics.py alongside neural_processes FittedCNP models, enabling
direct cross-system comparison on the same data arrays.
"""
from __future__ import annotations

import numpy as np

from dgraph.source.graph import Graph
from dgraph.examples.vol_smiles.source.nodes import SurfaceNode


class DGraphSurfacePredictor:
    """
    Adapts a fitted Graph (SurfaceNode → SurfaceState) to the SurfacePredictor
    interface defined in shared/surface_predictor.py.

    predict() ignores obs_* entirely — the parametric fit is already baked into
    the graph.  qry_feats[..., 0] = log-moneyness k, qry_feats[..., 1] = maturity T.
    asset_ids must be 0-based indices into `underlyings`.
    """

    def __init__(self, graph: Graph, underlyings: list[str]) -> None:
        """TODO.

        Args:
            graph: TODO.
            underlyings: TODO.
        """
        self.graph = graph
        self._nodes = [SurfaceNode(u) for u in underlyings]

    def predict(
        self,
        obs_feats: np.ndarray,  # ignored
        obs_aids:  np.ndarray,  # ignored
        obs_tgts:  np.ndarray,  # ignored
        qry_feats: np.ndarray,  # (B, Q, Q_dim)  [..., 0]=k  [..., 1]=T
        qry_aids:  np.ndarray,  # (B, Q)
    ) -> np.ndarray:            # (B, Q)
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
        B, Q = qry_aids.shape
        out = np.zeros((B, Q), dtype=np.float64)
        for a_idx, node in enumerate(self._nodes):
            if node not in self.graph.nodes:
                continue
            state = self.graph.get(node)
            for b in range(B):
                mask = qry_aids[b] == a_idx
                if not mask.any():
                    continue
                k = qry_feats[b, mask, 0].astype(float)
                T = qry_feats[b, mask, 1].astype(float)
                out[b, mask] = state.implied_vol(k, T)
        return out
