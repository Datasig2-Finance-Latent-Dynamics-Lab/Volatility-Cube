import numpy as np

from DependencyGraph.source.graph import Graph
from DependencyGraph.source.observation import ObservationSet
from DependencyGraph.losses.data import DataLoss
from ..source.nodes import CurveNode


class VolDataLoss(DataLoss):
    """
    Weighted MSE between fitted implied vol and observed implied vol.

    Each Observation.data is expected to be a (k, iv) tuple where
    k is log-moneyness and iv is the observed implied volatility.
    T is read from the CurveNode so it never enters the state vector.
    """

    def __init__(self) -> None:
        self._obs_cache: dict = {}

    def _build_node_arrays(self, observations: ObservationSet) -> dict:
        """
        Per-node numpy arrays are built once per ObservationSet and cached
        so they are not rebuilt on every optimizer evaluation.
        """
        cache: dict[CurveNode, tuple] = {}
        for nid, obs_list in observations._by_node.items():
            if not isinstance(nid, CurveNode):
                continue
            cache[nid] = (
                np.array([o.data[0] for o in obs_list]),
                np.array([o.data[1] for o in obs_list]),
                np.array([o.weight  for o in obs_list]),
            )
        return cache

    def _get_arrays(self, observations: ObservationSet) -> dict:
        obs_id = id(observations)
        if obs_id not in self._obs_cache:
            self._obs_cache[obs_id] = self._build_node_arrays(observations)
        return self._obs_cache[obs_id]

    def __call__(self, graph: Graph, observations: ObservationSet) -> float:
        node_arrays = self._get_arrays(observations)
        total = 0.0
        total_weight = 0.0
        for nid in graph.node_ids():
            if nid not in node_arrays:
                continue
            ks, iv_obs, weights = node_arrays[nid]
            iv_fit = graph.get(nid).implied_vol(ks)
            total        += float(np.dot(weights, (iv_fit - iv_obs) ** 2))
            total_weight += float(weights.sum())
        return total / total_weight if total_weight > 0 else 0.0

    def metrics(self, graph: Graph, observations: ObservationSet) -> dict[str, float]:
        """Returns weighted MSE, MAE, and MAPE over all nodes."""
        node_arrays = self._get_arrays(observations)
        sq = abs_ = rel = w_total = 0.0
        for nid in graph.node_ids():
            if nid not in node_arrays:
                continue
            ks, iv_obs, weights = node_arrays[nid]
            iv_fit = graph.get(nid).implied_vol(ks)
            err = iv_fit - iv_obs
            sq     += float(np.dot(weights, err ** 2))
            abs_   += float(np.dot(weights, np.abs(err)))
            rel    += float(np.dot(weights, np.abs(err) / np.maximum(iv_obs, 1e-10)))
            w_total += float(weights.sum())
        if w_total <= 0:
            return {"mse": 0.0, "mae": 0.0, "mape": 0.0}
        return {"mse": sq / w_total, "mae": abs_ / w_total, "mape": rel / w_total}


