from typing import Union

import numpy as np

from ..operator import UnaryOperator
from ..variable import Variable


class Power(UnaryOperator):
    def __init__(self,
                 x: Variable,
                 p: float) -> None:
        super().__init__(x)
        self.p = p

    def backward(self,
                 grad: float) -> None:
        self.x.backward(grad * self.x.value ** (self.p - 1) * self.p)


class Exponentation(UnaryOperator):
    def __init__(self,
                 c: float,
                 x: Variable) -> None:
        super().__init__(x)
        self.c = c

    def backward(self,
                 grad: float) -> None:
        self.x.backward(grad * np.log(self.c) * self.c ** self.x.value)

def variable_pow(self: Variable, p: float) -> Variable:
    result = self.value ** p
    op = Power(self, p)
    return Variable(result, op)


def variable_rpow(self: Variable, c: float) -> Variable:
    result = c ** self.value
    op = Exponentation(c, self)
    return Variable(result, op)
