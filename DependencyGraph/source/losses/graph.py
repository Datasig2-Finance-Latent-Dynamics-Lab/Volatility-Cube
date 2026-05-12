from abc import ABC, abstractmethod

import numpy as np

from ..core.graph import Graph


class GraphLoss(ABC):
    """
    Abstract data class for graph losses across dependencies.
    """

    @abstractmethod
    def __call__(self, graph: Graph, rolled_prior: Graph) -> float:
        ...


class L2DependencyGraphLoss(GraphLoss):
    """
    Very simple graph loss.

    GL = sum_i,j w_i,j * || delta_theta_i - delta_theta_j ||^2
    """

    def __init__(self):
        self._edge_cache: dict = {}

    def _build_cache(self, graph: Graph, node_ids: list):
        """
        Builds a cache of source target nodes and weights. This is only used for speed
        since it saves having to check for edges that are None.
        """
        node_index = {nid: i for i, nid in enumerate(node_ids)}

        raw_edges = graph.dependencies.edges()
        edge_triples = [
            (node_index[s], node_index[t], v)
            for s, t, v in raw_edges
            if s in node_index and t in node_index
        ]

        if not edge_triples:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp), np.array([])
        
        src_idx = np.array([e[0] for e in edge_triples], dtype=np.intp)
        tgt_idx = np.array([e[1] for e in edge_triples], dtype=np.intp)
        ws      = np.array([e[2] for e in edge_triples], dtype=float)
        return src_idx, tgt_idx, ws

    def __call__(self, graph: Graph, rolled_prior: Graph) -> float:
        node_ids = graph.node_ids()
        cache_key = (id(graph.dependencies), tuple(node_ids))

        if cache_key not in self._edge_cache:
            self._edge_cache[cache_key] = self._build_cache(graph, node_ids)

        src_idx, tgt_idx, ws = self._edge_cache[cache_key]
        if src_idx.size == 0:
            return 0.0

        deltas = []
        for nid in node_ids:
            theta_new = graph.get(nid).parameters()
            prior_state = rolled_prior.nodes.get(nid)
            theta_rolled = prior_state.parameters() if prior_state is not None else np.zeros_like(theta_new)
            deltas.append(theta_new - theta_rolled)

        delta_matrix = np.stack(deltas)
        diffs = delta_matrix[src_idx] - delta_matrix[tgt_idx]
        return float(np.dot(ws, np.mean(diffs ** 2, axis=1)))
