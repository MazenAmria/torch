from typing import List

from .node import Node
from .variable import Variable


class UnaryOperator(Node):
    def __init__(self, x: Variable) -> None:
        self.x = x

    def parameters(self) -> List[Variable]:
        return self.x.parameters()


class BinaryOperator(Node):
    def __init__(self, a: Variable, b: Variable) -> None:
        self.a = a
        self.b = b

    def parameters(self) -> List[Variable]:
        return self.a.parameters() + self.b.parameters()
