from numbers import Number

import numpy as np

from ..operator import UnaryOperator
from ..variable import Variable


class NaturalLogarithm(UnaryOperator):
    def __init__(self, x: Variable) -> None:
        super().__init__(x)

    def backward(self,
                 grad: Number) -> None:
        self.x.backward(grad / self.x.value)


def variable_log(self: Variable, other: Number = None) -> Variable:
        result = np.log(self.value)
        op = NaturalLogarithm(self)
        
        if other is None:
            return Variable(result, op)
        else:
            return Variable(result, op) / np.log(other)
