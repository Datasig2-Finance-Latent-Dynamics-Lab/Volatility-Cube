import time

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

    Note: for separable losses (no GraphLoss) prefer SeparableGraphUpdater,
    which solves one n_params-D problem per node instead of one (n_nodes *
    n_params)-D joint problem which is much faster.

    If loss.roller is set and prior_graph is provided, node precisions in the
    returned graph are initialised from the rolled prior (i.e. decayed) rather
    than carried over from the initial graph.  If precision_gain is also set,
    update_node_precision is applied after fitting to accumulate information
    from the new observations.
    """

    def __init__(
        self,
        loss: CombinedLoss,
        roller: Roller | None = None,
        bounds=None,
        method: str = "L-BFGS-B",
        precision_gain: float | None = None,
        verbose: bool = False,
    ):
        self.loss = loss
        self.roller = roller
        self.bounds = bounds
        self.method = method
        self.precision_gain = precision_gain
        self.verbose = verbose

    def update(
        self,
        graph: Graph,
        observations: ObservationSet,
        prior_graph: Graph | None = None,
    ) -> Graph:
        # get time increment
        dt: float | None = (
            (observations.date - prior_graph.date).days / 365
            if prior_graph is not None
            else None
        )

        # roll prior
        rolled: Graph | None = None
        if self.roller is not None and prior_graph is not None and dt is not None:
            rolled = self.roller.roll(prior_graph, dt)

        # minimize loss
        x0 = graph.to_vector()
        n_nodes = len(list(graph.node_ids()))

        if self.verbose:
            t0 = time.time()
            iters = [0]
            last_loss = [float("inf")]

            def objective(v: np.ndarray) -> float:
                val = self.loss(graph.from_vector(v), observations, rolled)
                last_loss[0] = val
                return val

            def callback(_):
                iters[0] += 1
                elapsed = time.time() - t0
                print(
                    f"  iter {iters[0]:4d} | loss {last_loss[0]:.6f}"
                    f" | {elapsed:.1f}s | {n_nodes} nodes",
                    end="\r",
                )
        else:
            def objective(v: np.ndarray) -> float:
                return self.loss(graph.from_vector(v), observations, rolled)
            callback = None

        bounds = self.bounds if self.bounds is not None else graph.parameter_bounds()
        result = minimize(objective, x0, method=self.method, bounds=bounds, callback=callback)

        if self.verbose:
            elapsed = time.time() - t0
            print(
                f"  done  {iters[0]:4d} iters | loss {result.fun:.6f}"
                f" | {elapsed:.1f}s | {n_nodes} nodes"
            )
        fitted = graph.from_vector(result.x)

        # set precision to that of the prior
        if rolled is not None:
            new_nodes = {
                nid: fitted.get(nid).with_precision(rolled.get(nid).precision)
                if nid in rolled.nodes else fitted.get(nid)
                for nid in fitted.node_ids()
            }
            fitted = Graph(fitted.date, new_nodes, fitted.edges)

        # update precision based on observations
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
    """
    Update node precision after fitting based on total observation weight.

    For each node with observations:
        P_new = P_current + c · (Σ weights) · I
    """
    new_nodes: dict = {}

    for nid in graph.node_ids():
        state = graph.get(nid)
        obs_list = observations.for_node(nid) # get observations for each node.

        if obs_list:
            total_weight = sum(o.weight for o in obs_list)
            p = state.precision
            if isinstance(p, (int, float)):
                new_precision = p + c * total_weight
            else:
                new_precision = p + c * total_weight * np.eye(state.n_params) # Assumes that precision only changes diagonal entries.
            new_nodes[nid] = state.with_precision(new_precision)
        else:
            new_nodes[nid] = state

    return Graph(graph.date, new_nodes, graph.edges)
