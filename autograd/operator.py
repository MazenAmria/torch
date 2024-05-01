from typing import List, Union
from numbers import Number

from .node import Node
from .variable import Variable


class UnaryOperator(Node):
    def __init__(self, x: Variable) -> None:
        self.x = x

    def parameters(self) -> List[Variable]:
        return self.x.parameters()


class BinaryOperator(Node):
    def __init__(self, a: Union[Variable, Number], b: Union[Variable, Number]) -> None:
        self.a = a
        self.b = b

    def parameters(self) -> List[Variable]:
        params = []
        if isinstance(self.a, Variable):
            params += self.a.parameters()
        if isinstance(self.b, Variable):
            params += self.b.parameters()
        return params
