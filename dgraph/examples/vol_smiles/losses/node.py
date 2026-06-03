from collections import defaultdict
import numpy as np

from dgraph.source.graph import Graph
from dgraph.losses.node import NodeLoss
from ..source.curves.svi import SviJWState
from ..source.curves.bspline import BSplineState
from ..source.nodes import CurveNode


class SviJWNodeLoss(NodeLoss):
    """
    Smooth soft-constraint penalty for SviJWState parameters.

    Two constraints that the box bounds alone cannot enforce:

      1. v_tilde <= v     (minimum variance must not exceed ATM variance)
      2. -p < psi < c     (equivalent to ``|d|`` < 1, i.e. valid to_raw() conversion)

    Both are penalised quadratically so the optimizer retains gradient
    information instead of hitting a flat 1e10 wall.  Non-JW nodes are
    silently skipped.
    """

    def __init__(self, weight: float = 1e4):
        self.weight = weight

    def __call__(self, graph: Graph) -> float:
        total = 0.0
        for nid in graph.node_ids():
            state = graph.get(nid)
            if not isinstance(state, SviJWState):
                continue
            params = [state.v, state.psi, state.p, state.c, state.v_tilde]
            if any(np.isnan(x) for x in params):
                total += 1e6
                continue
            # v_tilde <= v
            total += max(state.v_tilde - state.v, 0.0) ** 2
            # -p < psi  (violated when psi <= -p)
            total += max(-state.p - state.psi, 0.0) ** 2
            # psi < c   (violated when psi >= c)
            total += max(state.psi - state.c, 0.0) ** 2
        return self.weight * total


class BSplineNALoss(NodeLoss):
    """
    Butterfly no-arbitrage penalty for BSplineState via Durrleman's condition.

    Evaluates g(k) = (1 - k·w'/(2w))² - w'²/4·(1/w + 1/4) + w''/2 on a
    uniform grid and penalises regions where g(k) < 0 (non-negative density
    violated).  Derivatives are computed via numpy.gradient (finite differences).
    """

    def __init__(self, weight: float = 1e4, n_grid: int = 50):
        self.weight = weight
        self.n_grid = n_grid

    def __call__(self, graph: Graph) -> float:
        total = 0.0
        for nid in graph.node_ids():
            state = graph.get(nid)
            if not isinstance(state, BSplineState):
                continue
            k_lo = float(state.knots[state.degree])
            k_hi = float(state.knots[-state.degree - 1])
            if k_hi - k_lo < 1e-10:
                continue
            k = np.linspace(k_lo, k_hi, self.n_grid)
            w = state.total_variance(k)
            if not np.all(np.isfinite(w)):
                total += 1.0
                continue

            dk = k[1] - k[0]
            wp = np.gradient(w, dk)
            wpp = np.gradient(wp, dk)

            w_safe = np.maximum(w, 1e-8)
            g = (1.0 - k * wp / (2.0 * w_safe)) ** 2 \
                - wp ** 2 / 4.0 * (1.0 / w_safe + 0.25) \
                + wpp / 2.0

            violations = np.minimum(g, 0.0)
            total += float(np.sum(violations ** 2))

        return self.weight * total


class CalendarSpreadPenalty:
    """
    Soft calendar-spread no-arbitrage penalty applied to a whole graph.

    For each pair of consecutive maturities on the same underlying,
    penalises any strike where total variance decreases with time::

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
