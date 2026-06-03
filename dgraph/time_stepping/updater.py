from scipy.optimize import minimize
import numpy as np

from ..source.graph import Graph
from ..source.observation import ObservationSet
from ..losses.combined import CombinedLoss
from .roller import Roller


class GraphUpdater:
    """
    Fits the full graph in a single joint optimisation.

    The graph's to_vector / from_vector methods handle flattening and
    reconstruction.

    For separable losses (no GraphLoss) prefer SeparableGraphUpdater,
    which solves one n_params-D problem per node instead of one (n_nodes *
    n_params)-D joint problem which is much faster.
    """

    def __init__(
        self,
        loss: CombinedLoss,
        roller: Roller | None = None,
        bounds=None,
        method: str = "L-BFGS-B",
        precision_gain: float | None = None,
    ):
        """
        Args:
            loss: Combined loss function.
            roller: Method to roll priors.
            bounds: Bounds of the state parameters to optimize.
            method: Method name to minimize.
            precision_gain: Precision gained from observations at nodes.
        """
        self.loss = loss
        self.roller = roller
        self.bounds = bounds
        self.method = method
        self.precision_gain = precision_gain

    def update(
        self,
        graph: Graph,
        observations: ObservationSet,
        prior_graph: Graph | None = None,
    ) -> Graph:
        """TODO.

        Args:
            graph: Graph to update.
            observations: New observations.
            prior_graph: Prior graph.

        Returns:
            Graph
        """
        dt: float | None = (
            (observations.date - prior_graph.date).days / 365
            if prior_graph is not None
            else None
        )

        rolled: Graph | None = None
        if self.roller is not None and prior_graph is not None and dt is not None:
            rolled = self.roller.roll(prior_graph, dt)

        x0 = graph.to_vector()

        def objective(v: np.ndarray) -> float:
            return self.loss(graph.from_vector(v), observations, rolled)

        bounds = self.bounds if self.bounds is not None else graph.parameter_bounds()
        result = minimize(objective, x0, method=self.method, bounds=bounds)
        fitted = graph.from_vector(result.x)

        if rolled is not None:
            new_nodes = {
                nid: fitted.get(nid).with_precision(rolled.get(nid).precision)
                if nid in rolled.nodes else fitted.get(nid)
                for nid in fitted.node_ids()
            }
            fitted = Graph(fitted.date, new_nodes, fitted.edges)

        if self.precision_gain is not None:
            fitted = update_node_precision(fitted, observations, self.precision_gain)

        return fitted


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

    Precision rolling and accumulation follow the same logic as GraphUpdater.
    """

    def __init__(
        self,
        loss: CombinedLoss,
        roller: Roller | None = None,
        method: str = "L-BFGS-B",
        precision_gain: float | None = None,
    ):
        """TODO.

        Args:
            loss: Combined loss function.
            roller: Method to roll priors.
            method: Method name to minimize.
            precision_gain: Precision gained from observations at nodes.
        """
        self.loss = loss
        self.roller = roller
        self.method = method
        self.precision_gain = precision_gain

    def update(
        self,
        graph: Graph,
        observations: ObservationSet,
        prior_graph: Graph | None = None,
    ) -> Graph:
        """TODO.

        Args:
            graph: TODO.
            observations: TODO.
            prior_graph: TODO.

        Returns:
            TODO.
        """
        dt: float | None = (
            (observations.date - prior_graph.date).days / 365
            if prior_graph is not None
            else None
        )

        rolled: Graph | None = None
        if self.roller is not None and prior_graph is not None and dt is not None:
            rolled = self.roller.roll(prior_graph, dt)

        new_nodes = dict(graph.nodes)

        for nid in graph.node_ids():
            state = graph.get(nid)
            x0     = state.parameters()
            bounds = state.bounds()

            def objective(params: np.ndarray, _nid=nid, _state=state) -> float:
                mini = Graph(graph.date, {_nid: _state.from_parameters(params)}, graph.edges)
                return self.loss(mini, observations, rolled)

            result = minimize(objective, x0, method=self.method, bounds=bounds)
            fitted_state = state.from_parameters(result.x)

            if rolled is not None and nid in rolled.nodes:
                fitted_state = fitted_state.with_precision(rolled.get(nid).precision)

            new_nodes[nid] = fitted_state

        fitted = Graph(graph.date, new_nodes, graph.edges)

        if self.precision_gain is not None:
            fitted = update_node_precision(fitted, observations, self.precision_gain)

        return fitted


def update_node_precision(
    graph: Graph,
    observations: ObservationSet,
    c: float,
) -> Graph:
    """Updates precision at each node based on amount of observations at the given node.

    Args:
        graph: Graph to update precisions.
        observations: New observations.
        c: How much precision gained from each observation.

    Returns:
        Graph
    """
    new_nodes: dict = {}

    for nid in graph.node_ids():
        state = graph.get(nid)
        obs_list = observations.for_node(nid)

        if obs_list:
            total_weight = sum(o.weight for o in obs_list)
            p = state.precision
            if isinstance(p, (int, float)):
                new_precision = p + c * total_weight
            else:
                new_precision = p + c * total_weight * np.eye(state.n_params)
            new_nodes[nid] = state.with_precision(new_precision)
        else:
            new_nodes[nid] = state

    return Graph(graph.date, new_nodes, graph.edges)
