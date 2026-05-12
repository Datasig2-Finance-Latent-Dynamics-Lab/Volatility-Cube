from abc import ABC, abstractmethod

from ..core.node import NodeId
from ..core.state import State


class NodeLoss(ABC):
    """
    Abstract class for other type of losses involving only nodes.

    For example penalizing NA.
    """

    @abstractmethod
    def __call__(self, node_id: NodeId, state: State) -> float:
        ...

# TODO: in examples.vol_smiles add NA node loss.
