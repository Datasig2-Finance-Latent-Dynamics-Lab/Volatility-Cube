from ..core.graph import Graph
from ..distances.graph import GraphDistance


class TemporalLoss:
    """
    Abstract data class for temporal loss, which assumes that the loss is
    always some distance between the graphs.
    """

    def __init__(self, graph_distance: GraphDistance):
        self.graph_distance = graph_distance

    def __call__(self, graph: Graph, rolled_prior: Graph) -> float:
        return self.graph_distance(graph, rolled_prior)
