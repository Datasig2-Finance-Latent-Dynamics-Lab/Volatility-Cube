from ..core.graph import Graph
from ..core.observation import ObservationSet
from ..core.roller import Roller
from .data import DataLoss
from .temporal import TemporalLoss
from .graph import GraphLoss
from .node import NodeLoss


class CombinedLoss:
    """
    Abstract data class for a combines loss. The final formula is:

    L(G) = lambda_data     * data_loss(G, O) +
           lambda_temporal * temporal_loss(G, R(G)) +
           lambda_graph    * graph_loss(G - R(G)) +
           lambda_node     * node_loss(G) +

    Uses a rolled cache to avoid rerolling on every optimization step.
    """

    def __init__(
        self,
        data_loss: DataLoss | None = None,
        temporal_loss: TemporalLoss | None = None,
        graph_loss: GraphLoss | None = None,
        node_loss: NodeLoss | None = None,
        roller: Roller | None = None,
        lambda_data: float = 1.0,
        lambda_temporal: float = 1.0,
        lambda_graph: float = 1.0,
        lambda_node: float = 1.0,
    ):
        self.data_loss = data_loss
        self.temporal_loss = temporal_loss
        self.graph_loss = graph_loss
        self.node_loss = node_loss
        self.roller = roller
        self.lambda_data = lambda_data
        self.lambda_temporal = lambda_temporal
        self.lambda_graph = lambda_graph
        self.lambda_node = lambda_node
        self._roll_cache_key: tuple | None = None
        self._rolled_prior: Graph | None = None

    def __call__(
        self,
        graph: Graph,
        observations: ObservationSet,
        prior_graph: Graph | None = None,
        dt: int | None = None,
    ) -> float:
        
        rolled_prior: Graph | None = None
        if self.roller is not None and prior_graph is not None and dt is not None:
            key = (id(prior_graph), dt)
            if key != self._roll_cache_key:
                self._roll_cache_key = key
                self._rolled_prior = self.roller.roll(prior_graph, dt)
            rolled_prior = self._rolled_prior

        total = 0.0

        if self.data_loss is not None:
            total += self.lambda_data * self.data_loss(graph, observations)

        if self.temporal_loss is not None and rolled_prior is not None:
            total += self.lambda_temporal * self.temporal_loss(graph, rolled_prior)

        if self.graph_loss is not None and rolled_prior is not None:
            total += self.lambda_graph * self.graph_loss(graph, rolled_prior)

        if self.node_loss is not None:
            for nid in graph.node_ids():
                total += self.lambda_node * self.node_loss(nid, graph.get(nid))

        return total
