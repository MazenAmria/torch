from abc import ABC, abstractmethod
from typing import List
from numbers import Number


class Node(ABC):
    @abstractmethod
    def backward(self,
                 grad: Number = 1.0) -> None:
        pass
    
    @abstractmethod
    def parameters(self) -> List['Variable']:
        pass
