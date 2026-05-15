from dataclasses import dataclass

import pandas as pd

from DependencyGraph.source.node import NodeId


@dataclass(frozen=True)
class CurveNode(NodeId):
    """
    Node representing a single parametric curve, indexed by (underlying, expiry date).

    Identity is fully determined by (underlying, expiry).  Time-to-expiry T
    is stored in the state, not the node, to avoid duplication.
    """

    underlying: str
    expiry: pd.Timestamp


@dataclass(frozen=True)
class SurfaceNode(NodeId):
    """Node representing a full surface, indexed by underlying only."""
    underlying: str
