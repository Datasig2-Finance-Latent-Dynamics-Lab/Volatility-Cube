from abc import ABC, abstractmethod


class NodeId(ABC):
    """
    Data class for node_id. This is a bit painful to use mainly due to problems with
    finite arithmetic precision (specifically with time to expiry). Which is why special
    hash and equality are needed.
    """

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...
