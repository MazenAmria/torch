from abc import ABC, abstractmethod
from numbers import Real


class Node(ABC):
    @abstractmethod
    def backward(self,
                 grad: Real = 1.0) -> None:
        pass
