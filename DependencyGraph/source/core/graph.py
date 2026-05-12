import numpy as np
import pandas as pd

from .node import NodeId
from .state import State
from .dependency import Dependencies


class Graph:
    """
    Class for a general graph.

    Args:
        date: date the graph represents.
        nodes: dictionary of nodes with states.
        dependencies: Dependencies object for edges.
        _node_order: fixed order of nodes for easy construction of graphs from vectors.
    """

    def __init__(
        self,
        date: pd.Timestamp,
        nodes: dict[NodeId, State],
        dependencies: Dependencies,
    ):
        self.date = date
        self.nodes = nodes
        self.dependencies = dependencies
        self._node_order: list[NodeId] = list(nodes.keys())

    def to_vector(self) -> np.ndarray:
        """
        Transforms graph nodes onto vector to be able to use minimize from scipy.
        """
        return np.concatenate(
            [self.nodes[nid].parameters() for nid in self._node_order]
        )

    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        """
        Returns bounds for each vector in the same order as to_vector. Similarly, it
        is used in minimize.
        """
        all_bounds = []
        for nid in self._node_order:
            b = self.nodes[nid].bounds()
            if b is None:
                return None
            all_bounds.extend(b)
        return all_bounds

    def from_vector(self, v: np.ndarray) -> "Graph":
        """
        Constructes a graph with the same structure from a list which assumes that state parameters
        are orderes as in _node_order.
        """
        new_nodes: dict[NodeId, State] = {}
        offset = 0
        for nid in self._node_order:
            state = self.nodes[nid]
            n = state.n_params
            new_nodes[nid] = state.from_parameters(v[offset : offset + n])
            offset += n
        return Graph(self.date, new_nodes, self.dependencies)

    def get(self, node_id: NodeId) -> State:
        return self.nodes[node_id]

    def node_ids(self) -> list[NodeId]:
        return list(self._node_order)


# TODO: Add weights or precisions on each node and each edge for loss calculations.