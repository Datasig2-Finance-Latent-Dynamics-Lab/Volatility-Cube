from __future__ import annotations

import numpy as np

from ..source.graph import Graph


class GraphLoss:
    """
    Sums the precision weighted residual norm over all directed edges:

        L = Σ_{(i,j)}  r_{i,j}^T  Λ_{i,j}  r_{i,j}

    where r = edge.residual(state_i, state_j, rolled_i, rolled_j)
    and   Λ = edge.precision  (scalar or matrix).

    Edges whose endpoints are missing from either graph are skipped.
    """

    def __call__(self, graph: Graph, rolled_prior: Graph) -> float:
        total = 0.0
        for (nid_i, nid_j), edge in graph.edges.items():
            try:
                state_i  = graph.get(nid_i)
                state_j  = graph.get(nid_j)
                rolled_i = rolled_prior.get(nid_i)
                rolled_j = rolled_prior.get(nid_j)
            except KeyError:
                continue
            r = edge.residual(state_i, state_j, rolled_i, rolled_j)
            p = edge.precision
            if isinstance(p, (int, float)):
                total += p * float(np.dot(r, r))
            else:
                total += float(r @ p @ r)
        return total
