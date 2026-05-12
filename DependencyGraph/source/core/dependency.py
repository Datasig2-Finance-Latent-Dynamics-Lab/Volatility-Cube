from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .node import NodeId

EdgeValue = Any

class Dependencies(ABC):
    """
    Store the dependencies/edges of a graph.
    """

    @abstractmethod
    def get_edge(self, source: NodeId, target: NodeId) -> EdgeValue:
        """Gets the edge with given source and target nodes."""
        ...

    def edges(self) -> list[tuple[NodeId, NodeId, EdgeValue]]:
        """Returns dictionary with all the edges as tuples (source, target, edge)."""
        raise NotImplementedError
    
    # TODO: This were added to leave open the possibility of learning dependencies or optimizing over them too.
    # Haven't thought too much about it.

    def parameters(self) -> np.ndarray | None:
        """Returns all parameters as an array."""
        return None

    def from_parameters(self, params: np.ndarray) -> "Dependencies":
        """Gets constructed from list of parameters."""
        raise NotImplementedError


class StaticDependencies(Dependencies):
    """
    Stores static scalar dependencies.

    Args:
        _edge_map: edges as a dictionary
        _edges: edges as a list.
    """

    def __init__(self, edges: dict[tuple[NodeId, NodeId], EdgeValue]):
        self._edge_map = edges
        self._edges = [(s, t, v) for (s, t), v in edges.items() if v is not None]

    def get_edge(self, source: NodeId, target: NodeId) -> EdgeValue:
        return self._edge_map.get((source, target), None)

    def edges(self) -> list[tuple[NodeId, NodeId, EdgeValue]]:
        return self._edges
