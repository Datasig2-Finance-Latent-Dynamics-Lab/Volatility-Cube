from collections import defaultdict

import numpy as np

from ....source.core.graph import Graph
from ....source.core.observation import ObservationSet
from ....source.losses.data import DataLoss
from ....source.losses.graph import GraphLoss
from ..nodes import CurveNode


class VolDataLoss(DataLoss):
    """
    Weighted MSE between fitted implied vol and observed implied vol.

    Each Observation.data is expected to be a (k, iv) tuple where
    k is log-moneyness and iv is the observed implied volatility.
    T is read from the CurveNode so it never enters the state vector.

    Per-node numpy arrays are built once per ObservationSet and cached
    so they are not reallocated on every optimizer evaluation.
    """

    def __init__(self) -> None:
        self._obs_cache: dict = {}

    def _build_node_arrays(self, observations: ObservationSet) -> dict:
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

    def __call__(self, graph: Graph, observations: ObservationSet) -> float:
        obs_id = id(observations)
        if obs_id not in self._obs_cache:
            self._obs_cache[obs_id] = self._build_node_arrays(observations)
        node_arrays = self._obs_cache[obs_id]

        total = 0.0
        total_weight = 0.0
        for nid in graph.node_ids():
            if nid not in node_arrays:
                continue
            ks, iv_obs, weights = node_arrays[nid]
            iv_fit = graph.get(nid).implied_vol(ks, nid.T)
            total        += float(np.dot(weights, (iv_fit - iv_obs) ** 2))
            total_weight += float(weights.sum())
        return total / total_weight if total_weight > 0 else 0.0


class CalendarSpreadPenalty(GraphLoss):
    """
    Soft calendar-spread no-arbitrage penalty.

    For each pair of consecutive maturities on the same underlying,
    penalises any strike where total variance decreases with time:
        penalty = sum_k max(w(k, T_i) - w(k, T_{i+1}), 0)^2   for T_i < T_{i+1}
    """

    def __init__(self, grid: np.ndarray):
        self.grid = grid

    def __call__(self, graph: Graph, rolled_prior: Graph) -> float:
        by_underlying: dict[str, list[CurveNode]] = defaultdict(list)
        for nid in graph.node_ids():
            if isinstance(nid, CurveNode):
                by_underlying[nid.underlying].append(nid)

        total = 0.0
        for nodes in by_underlying.values():
            nodes_sorted = sorted(nodes, key=lambda n: n.T)
            for n1, n2 in zip(nodes_sorted, nodes_sorted[1:]):
                w1 = graph.get(n1).total_variance(self.grid)
                w2 = graph.get(n2).total_variance(self.grid)
                violations = np.maximum(w1 - w2, 0.0)
                total += float(np.sum(violations ** 2))
        return total
