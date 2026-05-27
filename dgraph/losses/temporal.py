from abc import ABC, abstractmethod

import numpy as np

from ..source.graph import Graph
from ..source.state import State


# ---------------------------------------------------------------------------
# State distances (inlined from source/distances/state.py)
# ---------------------------------------------------------------------------

class StateDistance(ABC):
    @abstractmethod
    def __call__(self, s1: State, s2: State) -> float:
        ...


class L2ParameterDistance(StateDistance):
    """MSE between state parameter vectors, ignores precisions."""
    def __call__(self, s1: State, s2: State) -> float:
        diff = s1.parameters() - s2.parameters()
        return float(np.mean(diff ** 2))


class PrecisionWeightedL2Distance(StateDistance):
    """
    Precision-weighted quadratic distance:  r^T Λ r

    where r = s1.parameters() - s2.parameters()
    and   Λ = s2.precision  (scalar or n_params x n_params matrix).I

    s1 is the new candidate state; precision weights how tightly the
    temporal loss pulls it toward the rolled prior s2.

    Note: technically not a metric by standard mathematical definition.
    Not symmetric since it uses precision of the SECOND input.
    """

    def __call__(self, new: State, prior: State) -> float:
        r = new.parameters() - prior.parameters()
        p = prior.precision  # prior precision: how informative is the prior
        if isinstance(p, (int, float)):
            return float(p * np.dot(r, r))
        return float(r @ p @ r)


# TODO: Add distance based on L2 distance between functions of the state (e.g implied volatility)


# ---------------------------------------------------------------------------
# Graph distances (inlined from source/distances/graph.py)
# ---------------------------------------------------------------------------

class GraphDistance(ABC):
    """
    Abstract data class for distance between two graphs.
    """
    @abstractmethod
    def __call__(self, g1: Graph, g2: Graph) -> float:
        ...


class NodewiseGraphDistance(GraphDistance):

    """
    Calculates the distance bwteen two graphs as the sum of the distance
    between states with the same idndex.

    Args:
        state_distance: function to compute "distance" between two states
    """

    def __init__(self, state_distance: StateDistance):
        self.state_distance = state_distance

    def __call__(self, g1: Graph, g2: Graph) -> float:
        common = set(g1.node_ids()) & set(g2.node_ids())
        return sum(self.state_distance(g1.get(nid), g2.get(nid)) for nid in common)


# ---------------------------------------------------------------------------
# Temporal loss
# ---------------------------------------------------------------------------

class TemporalLoss:
    """
    Abstract data class for temporal loss, which assumes that the loss is
    always some distance between the graphs.
    """

    def __init__(self, graph_distance: GraphDistance):
        self.graph_distance = graph_distance

    def __call__(self, graph: Graph, rolled_prior: Graph) -> float:
        return self.graph_distance(graph, rolled_prior)
