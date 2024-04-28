from abc import ABC, abstractmethod
from typing import List


class Node(ABC):
    @abstractmethod
    def backward(self,
                 grad: float) -> None:
        pass
    
    @abstractmethod
    def parameters(self) -> List['Variable']:
        pass
