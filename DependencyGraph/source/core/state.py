from abc import ABC, abstractmethod
import numpy as np


class State(ABC):
    """
    Underlying state stored at each node.
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Returns amount of parameters"""
        ...

    @abstractmethod
    def parameters(self) -> np.ndarray:
        """Returns vector of parameters."""
        ...

    @abstractmethod
    def from_parameters(self, params: np.ndarray) -> "State":
        """Constructs state from parameters."""
        ...

    @abstractmethod
    def copy(self) -> "State":
        ...

    def bounds(self) -> list[tuple[float | None, float | None]] | None:
        """Returns list of asmissible bounds for each parameter (useful for optimizer)."""
        return None
