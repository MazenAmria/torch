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

    def parameters(self) -> List[Variable]:
        x = DummyFloatList()
        y = self.forward(x)
        return y.parameters()

    def backward(self, grad: float) -> None:
        self.out.backward(grad)


class DummyFloatList(list):
    def __getitem__(self, key: int) -> float:
        return 0.0
