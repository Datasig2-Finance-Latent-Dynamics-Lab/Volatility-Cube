from abc import ABC, abstractmethod
import numpy as np

from ..core.state import State


class StateDistance(ABC):
    @abstractmethod
    def __call__(self, s1: State, s2: State) -> float:
        ...


class L2ParameterDistance(StateDistance):
    """MSE between state parameter vectors."""
    def __call__(self, s1: State, s2: State) -> float:
        diff = s1.parameters() - s2.parameters()
        return float(np.mean(diff ** 2))

# TODO: Add distance with weight matrix as proposed by Alexandre.


# TODO: Add distance based on distance between functions of the state (e.g implied volatility)