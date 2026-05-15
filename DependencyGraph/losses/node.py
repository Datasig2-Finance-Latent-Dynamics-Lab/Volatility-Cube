from abc import ABC, abstractmethod

from ..source.graph import Graph


class NodeLoss(ABC):
    """
    Abstract class for other type of losses involving only
    the specific node states amd ignoring dependencies.
    """

    @abstractmethod
    def __call__(self, graph : Graph) -> float:
        ...

# TODO: in examples.vol_smiles add NA node loss.
