import numpy as np
import pandas as pd

from .node import NodeId
from .state import State
from .edge import EdgeState


class Graph:
    """
    A graph of parametric node states connected by directed edges.

    Attributes:
        date (pd:Timestamp): Time of the graph.
        nodes (dict[NodeId, State]): Stores the nodes states.
        edges (dict[tuple[NodeId, NodeId], EdgeState]): Stores the edge states.
        _node_order (list[NodeId]): Keeps an order of NodeIds for graph reconstruction.
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
        """Vectorizes the whole graph. Needed to use scipy's minimize.

        Returns:
            np.ndarray
        """
        return np.concatenate(
            [self.nodes[nid].parameters() for nid in self._node_order]
        )

    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        """Vectorizes all the bounds for parameters in appropiate order.

        Returns:
            list
        """
        all_bounds = []
        for nid in self._node_order:
            b = self.nodes[nid].bounds()
            if b is None:
                return None
            all_bounds.extend(b)
        return all_bounds

    def from_vector(self, v: np.ndarray) -> "Graph":
        """Reconstructs a graph with same nodes from a vector.

        Args:
            v (np.ndarray): Vector to reconstruct from

        Returns:
            Graph
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
        """Gets state at a node.

        Args:
            node_id

        Returns:
            State
        """
        return self.nodes[node_id]

    def get_edge(self, src: NodeId, tgt: NodeId) -> EdgeState | None:
        """Gets state at an edge.

        Args:
            src (NodeId): Id of the source of the edge.
            tgt (NodeId): Id of the target of the edge.

        Returns:
            EdgeState
        """
        return self.edges.get((src, tgt))

    def node_ids(self) -> list[NodeId]:
        return list(self._node_order)
