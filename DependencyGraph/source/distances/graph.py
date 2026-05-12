from abc import ABC, abstractmethod

from ..core.graph import Graph
from .state import StateDistance


class GraphDistance(ABC):
    """
    Abstract data class for distance between two graphs.
    """
    @abstractmethod
    def __call__(self, g1: Graph, g2: Graph) -> float:
        ...


class NodewiseGraphDistance(GraphDistance):
    
    """
    Calculates the distance bwteen two graphs as the sum of the distance
    between states with the same idndex.

    Args:
        state_distance: function to compute distance between two states
    """

    def __init__(self, state_distance: StateDistance):
        self.state_distance = state_distance

    def __call__(self, g1: Graph, g2: Graph) -> float:
        common = set(g1.node_ids()) & set(g2.node_ids())
        return sum(self.state_distance(g1.get(nid), g2.get(nid)) for nid in common)
