from typing import List, Iterator
from abc import abstractmethod

from autograd import Node, Variable


class Module(Node):
    def __call__(self, x: List[float]) -> Variable:
        self.out = self.forward(x)
        return self.out

    @abstractmethod
    def forward(self, x: List[float]) -> Variable:
        pass

    def parameters(self) -> Iterator[Variable]:
        for _, value in self.__dict__.items():
            if isinstance(value, Variable):
                yield value

    def backward(self, grad: float) -> None:
        self.out.backward(grad)
