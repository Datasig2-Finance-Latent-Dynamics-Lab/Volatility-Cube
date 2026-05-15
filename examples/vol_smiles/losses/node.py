from collections import defaultdict
import numpy as np

from DependencyGraph.source.graph import Graph
from DependencyGraph.source.node import NodeId
from DependencyGraph.source.state import State
from DependencyGraph.losses.node import NodeLoss
from ..source.curves.svi import SviJWState
from ..source.nodes import CurveNode


class SviJWNodeLoss(NodeLoss):
    """
    Smooth soft-constraint penalty for SviJWState parameters.

    Two constraints that the box bounds alone cannot enforce:

      1. v_tilde <= v     (minimum variance must not exceed ATM variance)
      2. -p < psi < c     (equivalent to |d| < 1, i.e. valid to_raw() conversion)

    Both are penalised quadratically so the optimizer retains gradient
    information instead of hitting a flat 1e10 wall.  Non-JW nodes are
    silently skipped.
    """

    def __init__(self, weight: float = 1e4):
        self.weight = weight

    def __call__(self, node_id: NodeId, state: State) -> float:

        if not isinstance(state, SviJWState):
            return 0.0

        penalty = 0.0
        # v_tilde <= v
        penalty += max(state.v_tilde - state.v, 0.0) ** 2
        # -p < psi  (violated when psi <= -p)
        penalty += max(-state.p - state.psi, 0.0) ** 2
        # psi < c   (violated when psi >= c)
        penalty += max(state.psi - state.c, 0.0) ** 2

        return self.weight * penalty


class CalendarSpreadPenalty:
    """
    Soft calendar-spread no-arbitrage penalty applied to a whole graph.

    For each pair of consecutive maturities on the same underlying,
    penalises any strike where total variance decreases with time:
        penalty = sum_k max(w(k, T_i) - w(k, T_{i+1}), 0)^2   for T_i < T_{i+1}

    Not a NodeLoss (which acts per-node); call this directly on the graph
    and add to the combined loss manually when needed.
    """

    def __init__(self, grid: np.ndarray):
        self.grid = grid

    def __call__(self, graph: Graph) -> float:
        by_underlying: dict[str, list[CurveNode]] = defaultdict(list)
        for nid in graph.node_ids():
            if isinstance(nid, CurveNode):
                by_underlying[nid.underlying].append(nid)

        total = 0.0
        for nodes in by_underlying.values():
            nodes_sorted = sorted(nodes, key=lambda n: n.expiry)
            for n1, n2 in zip(nodes_sorted, nodes_sorted[1:]):
                w1 = graph.get(n1).total_variance(self.grid)
                w2 = graph.get(n2).total_variance(self.grid)
                violations = np.maximum(w1 - w2, 0.0)
                total += float(np.sum(violations ** 2))
        return total
