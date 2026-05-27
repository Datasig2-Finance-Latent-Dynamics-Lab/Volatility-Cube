from ..source.graph import Graph
from ..source.observation import ObservationSet
from .data import DataLoss
from .temporal import TemporalLoss
from .graph import GraphLoss
from .node import NodeLoss


class CombinedLoss:
    """
    Evaluates the combined loss for a candidate graph given an observation set
    and an optional already-rolled prior graph.

    Rolling is the caller's responsibility (usually an updater).

    L(G) = lambda_data     * data_loss(G, O)
         + lambda_temporal * temporal_loss(G, rolled_prior)
         + lambda_graph    * graph_loss(G, rolled_prior)
         + lambda_node     * node_loss(G)
    """

    def __init__(
        self,
        data_loss: DataLoss | None = None,
        temporal_loss: TemporalLoss | None = None,
        graph_loss: GraphLoss | None = None,
        node_loss: NodeLoss | None = None,
        lambda_data: float = 1.0,
        lambda_temporal: float = 1.0,
        lambda_graph: float = 1.0,
        lambda_node: float = 1.0,
        verbose = False,
    ):
        self.data_loss = data_loss
        self.temporal_loss = temporal_loss
        self.graph_loss = graph_loss
        self.node_loss = node_loss
        self.lambda_data = lambda_data
        self.lambda_temporal = lambda_temporal
        self.lambda_graph = lambda_graph
        self.lambda_node = lambda_node
        self.verbose = verbose

    def __call__(
        self,
        graph: Graph,
        observations: ObservationSet,
        rolled_prior: Graph | None = None,
    ) -> float:
        """
        Computes total loss. Assumes that the given prior is already rolled.
        """
        total = 0.0

        if self.data_loss is not None:
            total += self.lambda_data * self.data_loss(graph, observations)
            if self.verbose: print(f"Data Loss is: {self.data_loss(graph, observations)}")


        if self.temporal_loss is not None and rolled_prior is not None:
            total += self.lambda_temporal * self.temporal_loss(graph, rolled_prior)
            if self.verbose: print(f"Temporal Loss is: {self.temporal_loss(graph, rolled_prior)}")

        if self.graph_loss is not None and rolled_prior is not None:
            total += self.lambda_graph * self.graph_loss(graph, rolled_prior)
            if self.verbose: print(f"Graph Loss is: {self.graph_loss(graph, rolled_prior)}")

        if self.node_loss is not None:
            total += self.lambda_node * self.node_loss(graph)
            if self.verbose: print(f"Node Loss is: {self.node_loss(graph)}")

        if self.verbose: print("\n")

        return total
