from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def backward(self,
                 grad: float) -> None:
        pass
