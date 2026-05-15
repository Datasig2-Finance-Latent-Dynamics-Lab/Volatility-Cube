import numpy as np
import pandas as pd

from DependencyGraph.source.graph import Graph
from DependencyGraph.time_stepping.roller import Roller


class SurfaceRoller(Roller):

    """
    Surface parameters live in (k, T) coordinates, so they do not
    need to be modified as calendar time advances.  Rolling only decays the
    node precision to reflect the increased uncertainty over time.
    """

    def roll(self, graph: Graph, dt: float) -> Graph:
        new_nodes = {
            nid: state.with_precision(state.precision * np.exp(-dt))
            for nid, state in graph.nodes.items()
        }
        return Graph(
            graph.date + pd.Timedelta(days=round(dt * 365)),
            new_nodes,
            graph.edges,
        )
