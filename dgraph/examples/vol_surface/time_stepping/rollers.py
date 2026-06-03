import pandas as pd

from dgraph.source.graph import Graph
from dgraph.time_stepping.roller import Roller, decay_precision


class SurfaceRoller(Roller):

    """
    Surface parameters live in (k, T) coordinates, so they do not
    need to be modified as calendar time advances.  Rolling only decays the
    node precision to reflect the increased uncertainty over time.
    """

    def roll(self, graph: Graph, dt: float) -> Graph:
        new_nodes = {nid: decay_precision(state, dt) for nid, state in graph.nodes.items()}
        return Graph(
            graph.date + pd.Timedelta(days=round(dt * 365)),
            new_nodes,
            graph.edges,
        )
