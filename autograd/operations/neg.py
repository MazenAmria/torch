from typing import Union

from ..operator import Operator
from ..variable import Variable


class Negation(Operator):
    def __init__(self,
                 a: Variable) -> None:
        self.a = a

    def backward(self,
                 grad: float) -> None:
        self.a.backward(-1.0 * grad)


def variable_neg(self) -> Variable:
    result = -1 * self.value
    op = Negation(self)
    return Variable(result, op)
