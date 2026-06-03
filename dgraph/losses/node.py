from abc import ABC, abstractmethod

from ..source.graph import Graph


class NodeLoss(ABC):
    """
    Abstract class for other type of losses involving only
    the specific node states amd ignoring dependencies. For
    example violation of NA conditions.
    """

    @abstractmethod
    def __call__(self, graph: Graph) -> float:
        """
        Args:
            graph (Graph)

        Returns:
            float
        """
        ...

