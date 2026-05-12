from dataclasses import dataclass, field

import pandas as pd

from ...source.core.node import NodeId


@dataclass(frozen=True)
class CurveNode(NodeId):
    """
    Node representing a single parametric curve, indexed by (underlying, expiry date).

    `expiry` is the fixed calendar expiry date and is the sole basis for node
    identity (hash and equality).  `T` is the current time-to-expiry in years;
    it is used for computations but deliberately excluded from __hash__ and
    __eq__ so that rolled nodes still match their corresponding observations
    without floating-point drift problems.
    """
    underlying: str
    expiry: pd.Timestamp
    T: float = field(hash=False, compare=False)

    def advance(self, dt: float) -> "CurveNode":
        """Return node id after dt years have elapsed (T decreases, expiry unchanged)."""
        return CurveNode(self.underlying, self.expiry, self.T - dt)


@dataclass(frozen=True)
class SurfaceNode(NodeId):
    """Node representing a full surface, indexed by underlying only."""
    underlying: str
