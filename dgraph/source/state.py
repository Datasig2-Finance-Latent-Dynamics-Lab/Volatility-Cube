from abc import ABC, abstractmethod
import numpy as np


class State(ABC):
    """Abstract node state class."""

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Amount of parameters stored by state."""
        ...

    @abstractmethod
    def parameters(self) -> np.ndarray:
        """Parameters stored by state.

        Returns:
            np.ndarray
        """
        ...

    @abstractmethod
    def from_parameters(self, params: np.ndarray) -> "State":
        """Builds the state from the parameters.

        Args:
            params (np.ndarray)

        Returns:
            State
        """
        ...

    @abstractmethod
    def copy(self) -> "State":
        ...

    def bounds(self) -> list[tuple[float | None, float | None]] | None:
        """Gets the bounds on the parameters.

        Returns:
            list
        """
        return None

    @property
    def precision(self) -> float | np.ndarray:
        """Gets the preicion of the node."""
        return 1.0

    def with_precision(self, new_precision: float | np.ndarray) -> "State":
        """Creates a copy of the state with different precision.

        Args:
            new_precision (float)

        Returns:
            State
        """
        new = self.copy()
        new.precision = new_precision
        return new
