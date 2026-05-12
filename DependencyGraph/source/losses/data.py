from abc import ABC, abstractmethod

from ..core.graph import Graph
from ..core.observation import ObservationSet


class DataLoss(ABC):
    """
    Measures how well the graph fits the new observations.
    """

    @abstractmethod
    def __call__(self, graph: Graph, observations: ObservationSet) -> float:
        ...

# as mentioned in observations this assumes correspond to some node.