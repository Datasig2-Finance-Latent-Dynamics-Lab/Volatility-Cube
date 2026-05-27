import numpy as np

from dgraph.source.graph import Graph
from dgraph.source.observation import ObservationSet
from dgraph.losses.data import DataLoss
from ..source.states.base import SurfaceState


class SurfaceDataLoss(DataLoss):
    """
    Weighted MSE between fitted and observed implied vol over the full surface.

    Each Observation.data is expected to be a (k, T, iv) tuple.
    """

    def __init__(self) -> None:
        self._obs_cache: dict = {}

    def _build_node_arrays(self, observations: ObservationSet) -> dict:
        cache: dict = {}
        for nid, obs_list in observations._by_node.items():
            ks      = np.array([o.data[0] for o in obs_list])
            Ts      = np.array([o.data[1] for o in obs_list])
            iv_obs  = np.array([o.data[2] for o in obs_list])
            weights = np.array([o.weight   for o in obs_list])
            cache[nid] = (ks, Ts, iv_obs, weights)
        return cache

    def __call__(self, graph: Graph, observations: ObservationSet) -> float:
        obs_id = id(observations)
        if obs_id not in self._obs_cache:
            self._obs_cache[obs_id] = self._build_node_arrays(observations)

        node_arrays = self._obs_cache[obs_id]
        total        = 0.0
        total_weight = 0.0

        for nid in graph.node_ids():
            if nid not in node_arrays:
                continue
            state = graph.get(nid)
            if not isinstance(state, SurfaceState):
                continue
            ks, Ts, iv_obs, weights = node_arrays[nid]
            iv_fit        = state.implied_vol(ks, Ts)
            total        += float(np.dot(weights, (iv_fit - iv_obs) ** 2))
            total_weight += float(weights.sum())

        return total / total_weight if total_weight > 0 else 0.0

    def metrics(self, graph: Graph, observations: ObservationSet) -> dict[str, float]:
        obs_id = id(observations)
        if obs_id not in self._obs_cache:
            self._obs_cache[obs_id] = self._build_node_arrays(observations)
        node_arrays = self._obs_cache[obs_id]
        sq = abs_ = rel = w_total = 0.0
        for nid in graph.node_ids():
            if nid not in node_arrays:
                continue
            state = graph.get(nid)
            if not isinstance(state, SurfaceState):
                continue
            ks, Ts, iv_obs, weights = node_arrays[nid]
            iv_fit = state.implied_vol(ks, Ts)
            err = iv_fit - iv_obs
            sq     += float(np.dot(weights, err ** 2))
            abs_   += float(np.dot(weights, np.abs(err)))
            rel    += float(np.dot(weights, np.abs(err) / np.maximum(iv_obs, 1e-10)))
            w_total += float(weights.sum())
        if w_total <= 0:
            return {"mse": 0.0, "mae": 0.0, "mape": 0.0}
        return {"mse": sq / w_total, "mae": abs_ / w_total, "mape": rel / w_total}
