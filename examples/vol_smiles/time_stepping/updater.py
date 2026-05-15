"""
Analytic updater for B-spline vol graphs.

Because B-splines are linear in their coefficients, the combined loss
(data + temporal + graph) is quadratic in the coefficient vector.  Quadratic
problems have a closed-form solution: one linear-system solve per update step,
no iterative optimizer needed.

All losses are computed in implied-vol space σ(k) directly, since BSplineState
coefficients now represent IV (not total variance).
"""

# Done by ClaudeCode, doing normal minimization from scipy is painfully slow as one would expect.

# TODO: Look more into this if I make states that are parametrized by a basis in general.

from __future__ import annotations

import numpy as np
import pandas as pd

from DependencyGraph.source.graph import Graph
from DependencyGraph.source.observation import ObservationSet
from DependencyGraph.source.edge import DeltaEdgeState
from ..source.nodes import CurveNode
from ..source.curves.bspline import BSplineState


class BSplineUpdater:
    """
    Fits the full B-spline graph by solving the normal equations of the
    joint quadratic loss in one step.

    Loss (implied-vol space)
    ------------------------
      L = lambda_data * sum_i  ||sqrt(W_i) (B_i c_i - iv_obs_i)||^2
        + lambda_temporal * sum_i  ||c_i - c_prior_i||^2
        + lambda_graph * sum_{(i,j)} weight_ij * ||c_i - c_j||^2

    Setting dL/dc_i = 0 gives a block-coupled linear system H c = b which
    is assembled and solved in one shot via numpy.linalg.solve.

    Parameters
    ----------
    lambda_data      : weight on the data fidelity term
    lambda_temporal  : weight on the temporal regularisation term
    lambda_graph     : weight on the cross-node graph regularisation term
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_temporal: float = 0.0,
        lambda_graph: float = 0.0,
        verbose = False,
    ):
        self.lambda_data = lambda_data
        self.lambda_temporal = lambda_temporal
        self.lambda_graph = lambda_graph

        # cache: (obs_id, node_id) -> (B, w_obs, W_diag)
        self._obs_cache: dict = {}

    # ------------------------------------------------------------------
    # Public API (same signature as GraphUpdater.update)
    # ------------------------------------------------------------------

    def update(
        self,
        graph: Graph,
        observations: ObservationSet,
        prior_graph: Graph | None = None,
    ) -> Graph:
        node_ids = [nid for nid in graph.node_ids() if isinstance(nid, CurveNode)]
        n = len(node_ids)
        if n == 0:
            return graph

        # Retrieve or build per-node basis arrays
        node_arrays = self._get_node_arrays(graph, observations)

        state0 = graph.get(node_ids[0])
        if not isinstance(state0, BSplineState):
            raise TypeError("BSplineUpdater requires BSplineState nodes")
        p = state0.n_params  # number of coefficients per node

        # Build block linear system H c = b (size N*p × N*p)
        H = np.zeros((n * p, n * p))
        b = np.zeros(n * p)

        node_index = {nid: i for i, nid in enumerate(node_ids)}

        # Precompute prior parameters once; used by both temporal and graph terms.
        prior_params: dict = {}
        if prior_graph is not None:
            for nid_ in node_ids:
                try:
                    prior_params[nid_] = prior_graph.get(nid_).parameters()
                except KeyError:
                    pass

        for i, nid in enumerate(node_ids):
            sl = slice(i * p, (i + 1) * p)
            state = graph.get(nid)

            # --- data term ---
            if nid in node_arrays and self.lambda_data > 0:
                B, w_obs, W_diag = node_arrays[nid]
                BtWB = B.T @ (W_diag[:, None] * B)
                BtWw = B.T @ (W_diag * w_obs)
                H[sl, sl] += 2.0 * self.lambda_data * BtWB
                b[sl]     += 2.0 * self.lambda_data * BtWw

            # --- temporal term ---
            if nid in prior_params and self.lambda_temporal > 0:
                c_prior_i = prior_params[nid]
                H[sl, sl] += 2.0 * self.lambda_temporal * np.eye(p)
                b[sl]     += 2.0 * self.lambda_temporal * c_prior_i

            # --- graph term (off-diagonal coupling, delta-based) ---
            # For each DeltaEdgeState (i→j): loss = λ (Δc_i − M Δc_j)^T P (Δc_i − M Δc_j)
            # Normal-equations contributions to node i's block:
            #   H[i,i] += 2λ P
            #   H[i,j] -= 2λ P M          (M=I when edge.matrix is None)
            #   b[i]   += 2λ P (c*_i − M c*_j)
            if self.lambda_graph > 0:
                for j, nid_j in enumerate(node_ids):
                    if i == j:
                        continue
                    edge = graph.edges.get((nid, nid_j))
                    if not isinstance(edge, DeltaEdgeState):
                        continue
                    sl_j = slice(j * p, (j + 1) * p)
                    prec = edge.precision
                    P = float(prec) * np.eye(p) if isinstance(prec, (int, float)) else prec
                    M = edge.matrix
                    PM = P if M is None else P @ M
                    H[sl, sl]   += 2.0 * self.lambda_graph * P
                    H[sl, sl_j] -= 2.0 * self.lambda_graph * PM
                    if nid in prior_params and nid_j in prior_params:
                        rhs_correction = prior_params[nid] - (prior_params[nid_j] if M is None else M @ prior_params[nid_j])
                        b[sl] += 2.0 * self.lambda_graph * (P @ rhs_correction)

        # Regularise the diagonal slightly to guarantee invertibility when
        # some nodes have no observations and no prior.
        H += 1e-10 * np.eye(n * p)

        c_flat = np.linalg.solve(H, b)

        # Reconstruct graph; use H[i,i] block as posterior precision for each node.
        # H is the Hessian of the quadratic loss, so its diagonal blocks are the
        # exact posterior precision matrices (data + temporal + graph contributions).
        # This makes precision differ across models and between masked/observed nodes.
        new_nodes = dict(graph.nodes)
        for i, nid in enumerate(node_ids):
            sl = slice(i * p, (i + 1) * p)
            new_nodes[nid] = graph.get(nid).from_parameters(c_flat[sl]).with_precision(H[sl, sl].copy())

        return Graph(observations.date, new_nodes, graph.edges)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _get_node_arrays(
        self, graph: Graph, observations: ObservationSet
    ) -> dict[CurveNode, tuple[np.ndarray, np.ndarray, np.ndarray]]:

        obs_id = id(observations)
        if obs_id in self._obs_cache:
            return self._obs_cache[obs_id]

        result: dict = {}
        for nid in graph.node_ids():
            if not isinstance(nid, CurveNode):
                continue
            obs_list = observations.for_node(nid)
            if not obs_list:
                continue
            state = graph.get(nid)
            if not isinstance(state, BSplineState):
                continue

            ks      = np.array([o.data[0] for o in obs_list])
            iv_obs  = np.array([o.data[1] for o in obs_list])
            weights = np.array([o.weight   for o in obs_list])

            # Dense design matrix (n_obs × n_coeffs); cached inside BSplineState
            B = state._basis(ks)

            result[nid] = (B, iv_obs, weights)

        self._obs_cache[obs_id] = result
        return result
