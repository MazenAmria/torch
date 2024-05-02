from typing import Union
from numbers import Real

from .node import Node
from .variable import Variable


class UnaryOperator(Node):
    def __init__(self, x: Variable) -> None:
        self.x = x


class BinaryOperator(Node):
    def __init__(self, a: Union[Variable, Real], b: Union[Variable, Real]) -> None:
        self.a = a
        self.b = b
