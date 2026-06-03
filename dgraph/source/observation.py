from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .node import NodeId


@dataclass
class Observation:
    """
    Dataclass for an observation.
    Has the current "limitation" that it must be associated with a node.

    Attributes:
        node_id (NodeId): id of the node corresponding to the observation.
        data: Data stored in the observation.
        weight (float): Precision of the observation.
    """

    node_id: NodeId
    data: Any
    weight: float = 1.0


class ObservationSet:
    """
    Set of observations. Observation data is kept abstract with the idea that it may be a general type.
    Different data losses would handle observations differently.
    """

    def __init__(self, observations: list[Observation], date: pd.Timestamp):
        """
        Args:
            observations (list): List of the observations.
            date (pd.Timestamp): Date of the observations.
        """
        self.observations = observations
        self.date = date
        self._by_node: dict[NodeId, list[Observation]] = {} # observations splitted by nodes.
        for o in observations:
            self._by_node.setdefault(o.node_id, []).append(o)

    def for_node(self, node_id: NodeId) -> list[Observation]:
        """Returns observations for the given node.

        Args:
            node_id (NodeId)

        Returns:
            list
        """
        return self._by_node.get(node_id, [])

    def __len__(self) -> int:
        """Gets amount of observations.

        Returns:
            int
        """
        return len(self.observations)
