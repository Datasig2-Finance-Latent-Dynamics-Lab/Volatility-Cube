from abc import ABC, abstractmethod


class NodeId(ABC):
    """Id for a node."""

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...
