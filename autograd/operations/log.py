from numbers import Real

import numpy as np

from ..operator import UnaryOperator
from ..variable import Variable


class NaturalLogarithm(UnaryOperator):
    def __init__(self, x: Variable) -> None:
        super().__init__(x)

    def backward(self,
                 grad: Real = 1.0) -> None:
        self.x.backward(grad / self.x.value)


def variable_log(self: Variable) -> Variable:
        result = np.log(self.value)
        op = NaturalLogarithm(self)
        
        return Variable(result, op)
