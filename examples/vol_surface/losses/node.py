from DependencyGraph.source.node import NodeId
from DependencyGraph.source.state import State
from DependencyGraph.source.graph import Graph
from DependencyGraph.losses.node import NodeLoss
from ..source.states.ssvi import SSVISurfaceState


class SSVINodeLoss(NodeLoss):
    """
    Soft no-arbitrage penalty for SSVISurfaceState.

    Enforces the Gatheral-Jacquier condition:  η · (1 + |ρ|) ≤ 2
    The box bounds keep γ ≤ 0.5 and v_0, v_inf, κ > 0, so this is
    the only inequality that the optimizer can violate in a non-trivial way.
    """

    def __init__(self, weight: float = 1e4):
        self.weight = weight # not really needed as it has the same purpose as lambda_node in combined loss
        # but it is a good multiplier so taht lambda_node is same order of other lambdas.

    def __call__(self, graph : Graph) -> float:

        total = 0.0

        for state in graph.nodes.values():

            if not isinstance(state, SSVISurfaceState):
                raise ValueError("SSVINodeLoss must be called on graph with node states SSVISurfaceState")
            
            violation = state.eta * (1.0 + abs(state.rho)) - 2.0
        
            total += self.weight * max(violation, 0.0) ** 2

        return total
