from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .state import State


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
        """
        Return the residual vector for this edge. If e:i -> j,
        Usually of the form r(dv_j,dv_i). Where dv means increment of state.
        """
        ...

    @property
    @abstractmethod
    def precision(self) -> float | np.ndarray:
        """
        Precision on the residual space. Scalar or pxp matrix.
        Almast alway a scalar. If it is a matrix it MUST match the dimensions
        of the output of residual.
        """
        ...


class DeltaEdgeState(EdgeState):
    """
    Specific edge state which gives residual: r = delta_i - M_i,j delta_j
    where M is a linear transformation, i.e. matrix.
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

        # Make the increments.
        delta_i = state_i.parameters() - rolled_i.parameters()
        delta_j = state_j.parameters() - rolled_j.parameters()

        if self.matrix is None:
            return delta_j - delta_i
        return delta_j - self.matrix @ delta_i
