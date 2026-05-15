from abc import ABC, abstractmethod
import numpy as np


class State(ABC):
    """
    Underlying state stored at each node.

    Do not confuse with edge state.
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

    @property
    def precision(self) -> float | np.ndarray:
        """
        Precision on the parameter space. Can either br a acalar or
        (n_params x n_params) matrix. Used by PrecisionWeightedL2Distance in the temporal loss.
        """
        return 1.0

    def with_precision(self, new_precision: float | np.ndarray) -> "State":
        """
        Return a copy of this state with updated precision.
        Contract: concrete subclasses must expose a mutable `precision` attribute
        (e.g. a dataclass field) so that `copy()` followed by assignment works.
        """
        new = self.copy()
        new.precision = new_precision
        return new
