from scipy.optimize import minimize
import numpy as np

from .core.graph import Graph
from .core.observation import ObservationSet
from .losses.combined import CombinedLoss


class GraphUpdater:
    """
    Fits the full graph in a single joint optimisation.

    The graph's to_vector / from_vector methods handle flattening and
    reconstruction so this class stays application-agnostic.

    Note: for separable losses (no GraphLoss) prefer SeparableGraphUpdater,
    which solves one n_params-D problem per node instead of one (n_nodes *
    n_params)-D joint problem which is much faster.
    """

    def __init__(
        self,
        loss: CombinedLoss,
        bounds=None,
        method: str = "L-BFGS-B",
    ):
        self.loss = loss
        self.bounds = bounds
        self.method = method

    def update(
        self,
        graph: Graph,
        observations: ObservationSet,
        prior_graph: Graph | None = None,
    ) -> Graph:
        dt: float | None = (
            float((observations.date - prior_graph.date).days)
            if prior_graph is not None
            else None
        )

        x0 = graph.to_vector()

        def objective(v: np.ndarray) -> float:
            candidate = graph.from_vector(v)
            return self.loss(candidate, observations, prior_graph, dt)

        bounds = self.bounds if self.bounds is not None else graph.parameter_bounds()
        result = minimize(objective, x0, method=self.method, bounds=bounds)

        return graph.from_vector(result.x)


class SeparableGraphUpdater:
    """
    Optimises each node independently — valid for separable loss functions,
    i.e. DataLoss + TemporalLoss without any GraphLoss coupling.

    Each node becomes its own small optimisation problem (n_params dimensions
    instead of n_nodes * n_params), which converges in far fewer iterations and
    evaluations.  The existing CombinedLoss is reused: passing a single-node
    mini-graph to it causes each loss component to evaluate only that node's
    contribution, which is correct whenever both DataLoss and TemporalLoss are
    sums over nodes.
    """

    def __init__(self, loss: CombinedLoss, method: str = "L-BFGS-B"):
        self.loss = loss
        self.method = method

    def update(
        self,
        graph: Graph,
        observations: ObservationSet,
        prior_graph: Graph | None = None,
    ) -> Graph:
        dt: float | None = (
            float((observations.date - prior_graph.date).days)
            if prior_graph is not None
            else None
        )

        new_nodes = dict(graph.nodes)

        for nid in graph.node_ids():
            state = graph.get(nid)
            x0     = state.parameters()
            bounds = state.bounds()

            def objective(params: np.ndarray, _nid=nid, _state=state) -> float:
                # Single-node mini-graph: only this node's params vary.
                # Losses that are sums-over-nodes naturally reduce to just
                # this node's contribution.
                mini = object.__new__(Graph)
                mini.date         = graph.date
                mini.nodes        = {_nid: _state.from_parameters(params)}
                mini.dependencies = graph.dependencies
                mini._node_order  = [_nid]
                return self.loss(mini, observations, prior_graph, dt)

            result   = minimize(objective, x0, method=self.method, bounds=bounds)
            new_nodes[nid] = state.from_parameters(result.x)

        return Graph(graph.date, new_nodes, graph.dependencies)
