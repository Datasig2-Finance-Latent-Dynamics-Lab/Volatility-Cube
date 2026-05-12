from abc import ABC, abstractmethod

from .graph import Graph


class Roller(ABC):
    """
    Abstract class for roller.
    """

    @abstractmethod
    def roll(self, graph: Graph, dt: float) -> Graph:
        """Rolls a graph nodewise a given time increment."""
        ...
