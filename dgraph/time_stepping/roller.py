from abc import ABC, abstractmethod

import numpy as np

from ..source.graph import Graph
from ..source.state import State


def decay_precision(state: State, dt: float) -> State:
    """Return a copy of state with precision decayed exponentially: Λ → Λ · exp(-dt)."""
    return state.with_precision(state.precision * np.exp(-dt))


class Roller(ABC):
    """Abstract roller: advance a graph by dt years."""

    @abstractmethod
    def roll(self, graph: Graph, dt: float) -> Graph:
        """
        Args:
            graph (Graph): Graph to be rolled.
            dt (float): Time increment to roll by.

        Returns:
            Graph
        """
        ...
