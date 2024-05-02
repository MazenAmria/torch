from typing import Union
from numbers import Real

import numpy as np

from ..operator import UnaryOperator, BinaryOperator
from ..variable import Variable


class Exponentiation(BinaryOperator):
    def __init__(self,
                 a: Union[Variable, Real],
                 b: Union[Variable, Real]) -> None:
        super().__init__(a, b)

    def backward(self,
                 grad: Real = 1.0) -> None:
        a_val = self.a.value if isinstance(self.a, Variable) else self.a
        b_val = self.b.value if isinstance(self.b, Variable) else self.b

        if isinstance(self.a, Variable):
            self.a.backward(grad * a_val ** (b_val - 1) * b_val)
        if isinstance(self.b, Variable):
            self.b.backward(grad * np.log(a_val) * a_val ** b_val)


def variable_pow(self: Variable, other: Union[Variable, Real]) -> Variable:
    if isinstance(other, Real):
        result = self.value ** other
    elif isinstance(other, Variable):
        result = self.value ** other.value
    else:
        raise TypeError(f"unsupported operand type(s) for **: '{self.__class__}' and '{type(other)}'")
    
    op = Exponentiation(self, other)
    return Variable(result, op)


def variable_rpow(self: Variable, other: Real) -> Variable:
    result = other ** self.value
    op = Exponentiation(other, self)
    return Variable(result, op)
