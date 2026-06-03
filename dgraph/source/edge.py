from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .state import State

# No EdgeId needed since an EdgeId is just a tuple (NodeId,NodeId)

class EdgeState(ABC):
    """
    Abstract class for an edge state.
    For a matrix precision A, the graph loss is r^t A r.
    """

    @abstractmethod
    def residual(
        self,
        state_i: State,
        state_j: State,
        rolled_i: State,
        rolled_j: State,
    ) -> np.ndarray:
        """Gets r from two nodes.

        Args:
            state_i (State): New State for node i.
            state_j (State): New State for node j.
            rolled_i (State): Rolled State for node i.
            rolled_j (State): Rolled State for node j.

        Returns:
            np.ndarray
        """
        ...

    @property
    @abstractmethod
    def precision(self) -> float | np.ndarray:
        """Gets the precision of the edge. A quantification of how accurate we believe
        our residual calculation is."""
        ...


class DeltaEdgeState(EdgeState):
    """
    Specific edge state which gives residual: r = delta_i - M_i,j delta_j
    where M is a linear transformation, stored as a matrix.

    Attributes:
        precision (float | np.ndarray): Precision of the edge
        matrix (np.darray): Matrix for the delta edge.
    """

    def __init__(
        self,
        precision: float | np.ndarray,
        matrix: np.ndarray | None = None,
    ):
        self._precision = precision
        self.matrix = matrix

    @property
    def precision(self) -> float | np.ndarray:
        return self._precision

    def residual(
        self,
        state_i: State,
        state_j: State,
        rolled_i: State,
        rolled_j: State,
    ) -> np.ndarray:
        """TODO.

        Args:
            state_i (State): New State for node i.
            state_j (State): New State for node j.
            rolled_i (State): Rolled State for node i.
            rolled_j (State): Rolled State for node j.

        Returns:
            np.ndarray
        """

        # Make the increments.
        delta_i = state_i.parameters() - rolled_i.parameters()
        delta_j = state_j.parameters() - rolled_j.parameters()

        if self.matrix is None:
            return delta_j - delta_i
        return delta_j - self.matrix @ delta_i
