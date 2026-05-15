import numpy as np
import pandas as pd

from .node import NodeId
from .state import State
from .edge import EdgeState


class Graph:
    """
    A graph of parametric node states connected by directed edges.

    nodes : dict[NodeId, State]                          — node states
    edges : dict[tuple[NodeId, NodeId], EdgeState]       — directed edge states
    """

    def __init__(
        self,
        date: pd.Timestamp,
        nodes: dict[NodeId, State],
        edges: dict[tuple[NodeId, NodeId], EdgeState],
    ):
        self.date = date
        self.nodes = nodes
        self.edges = edges
        self._node_order: list[NodeId] = list(nodes.keys())

    def to_vector(self) -> np.ndarray:
        """
        Turns the whole graph into a vector so that we can use scpy minimize.
        """
        return np.concatenate(
            [self.nodes[nid].parameters() for nid in self._node_order]
        )

    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        all_bounds = []
        for nid in self._node_order:
            b = self.nodes[nid].bounds()
            if b is None:
                return None
            all_bounds.extend(b)
        return all_bounds

    def from_vector(self, v: np.ndarray) -> "Graph":
        """
        Reconstructs a graph with the same nodes and types from a vector.
        """
        new_nodes: dict[NodeId, State] = {}
        offset = 0
        for nid in self._node_order:
            state = self.nodes[nid]
            n = state.n_params
            new_nodes[nid] = state.from_parameters(v[offset : offset + n])
            offset += n
        return Graph(self.date, new_nodes, self.edges)

    def get(self, node_id: NodeId) -> State:
        """
        Returns state at a specific node.
        """
        return self.nodes[node_id]

    def get_edge(self, src: NodeId, tgt: NodeId) -> EdgeState | None:
        """
        Returns edgestate at specific edge.
        """
        return self.edges.get((src, tgt))

    def node_ids(self) -> list[NodeId]:
        """
        Makes an ordered list of the nodes. nEeded for consistency.
        """
        return list(self._node_order)
