from typing import List
from abc import abstractmethod

from autograd import Node, Variable


class Module(Node):
    def __call__(self, x: List[float]) -> Variable:
        self.out = self.forward(x)
        return self.out

    @abstractmethod
    def forward(self, x: List[float]) -> Variable:
        pass

    @abstractmethod
    def params(self) -> List[Variable]:
        pass

    def backward(self, grad: float) -> None:
        self.out.backward(grad)
