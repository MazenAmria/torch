from typing import Union

from ..operator import UnaryOperator
from ..variable import Variable


class Negation(UnaryOperator):
    def __init__(self,
                 x: Variable) -> None:
        super().__init__(x)

    def backward(self,
                 grad: float) -> None:
        self.x.backward(-1.0 * grad)


def variable_neg(self) -> Variable:
    result = -1 * self.value
    op = Negation(self)
    return Variable(result, op)
