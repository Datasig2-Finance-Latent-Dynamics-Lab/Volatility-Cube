from abc import ABC, abstractmethod

import numpy as np

from ..source.graph import Graph
from ..source.observation import ObservationSet


class DataLoss(ABC):
    """Measures how well the graph fits the new observations."""

    @abstractmethod
    def _build_node_arrays(self, observations: ObservationSet) -> dict:
        """Splits observation into a dictionary with nodeId being the keys.

        Args:
            observations (ObservationSet)

        Returns:
            dict
        """

    @abstractmethod
    def _eval_node(self, state, node_obs: tuple):
        """Evaluate the fitted surface against observations for a single node.

        Args:
            state (State): Fitted state at this node.
            node_obs (tuple): Observation arrays for this node.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: ``(iv_fit, iv_obs, weights)``,
            all of shape ``(n_obs,)``. Return ``None`` to skip this node
            (e.g. if the state type is wrong).
        """
        ...

    def __call__(self, graph: Graph, observations: ObservationSet) -> float:
        """
        Args:
            graph (Graph)
            observations (ObservationSet)

        Returns:
            float
        """
        node_arrays = self._build_node_arrays(observations)
        total = total_weight = 0.0
        for nid in graph.node_ids():
            if nid not in node_arrays:
                continue
            result = self._eval_node(graph.get(nid), node_arrays[nid])
            if result is None:
                continue
            fit, obs, weights = result
            total        += float(np.dot(weights, (fit - obs) ** 2))
            total_weight += float(weights.sum())
        return total / total_weight if total_weight > 0 else 0.0

    def metrics(self, graph: Graph, observations: ObservationSet) -> dict[str, float]:
        """Returns mse, mae and mape for a given set of Observations.

        Args:
            graph (Graoh)
            observations (ObservationSet)

        Returns:
            dict
        """
        # TODO: Super easy to add rmse if needed.
        node_arrays = self._build_node_arrays(observations)
        sq = abs_ = rel = w_total = 0.0
        for nid in graph.node_ids():
            if nid not in node_arrays:
                continue
            result = self._eval_node(graph.get(nid), node_arrays[nid])
            if result is None:
                continue
            iv_fit, iv_obs, weights = result
            err     = iv_fit - iv_obs
            sq      += float(np.dot(weights, err ** 2))
            abs_    += float(np.dot(weights, np.abs(err)))
            rel     += float(np.dot(weights, np.abs(err) / np.maximum(iv_obs, 1e-10)))
            w_total += float(weights.sum())
        if w_total <= 0:
            return {"mse": 0.0, "mae": 0.0, "mape": 0.0}
        return {"mse": sq / w_total, "mae": abs_ / w_total, "mape": rel / w_total}
