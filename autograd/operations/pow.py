from typing import Union
from numbers import Number

import numpy as np

from ..operator import UnaryOperator, BinaryOperator
from ..variable import Variable


class Power(UnaryOperator):
    def __init__(self,
                 x: Variable,
                 p: Number) -> None:
        super().__init__(x)
        self.p = p

    def backward(self,
                 grad: Number) -> None:
        self.x.backward(grad * self.x.value ** (self.p - 1) * self.p)


class Exponentiation(UnaryOperator):
    def __init__(self,
                 c: Number,
                 x: Variable) -> None:
        super().__init__(x)
        self.c = c

    def backward(self,
                 grad: Number) -> None:
        self.x.backward(grad * np.log(self.c) * self.c ** self.x.value)


class VariableExponentiation(BinaryOperator):
    def __init__(self,
                 a: Variable,
                 b: Variable) -> None:
        super().__init__(a, b)

    def backward(self,
                 grad: Number) -> None:
        self.a.backward(grad * self.a.value ** (self.b.value - 1) * self.b.value)
        self.b.backward(grad * np.log(self.a.value) * self.a.value ** self.b.value)


def variable_pow(self: Variable, other: Union[Variable, Number]) -> Variable:
    if isinstance(other, Number):
        result = self.value ** other
        op = Power(self, other)
        return Variable(result, op)
    if isinstance(other, Variable):
        result = self.value ** other.value
        op = VariableExponentiation(self, other)
        return Variable(result, op)
    else:
        raise TypeError(f"unsupported operand type(s) for **: '{self.__class__}' and '{type(other)}'")


def variable_rpow(self: Variable, other: Number) -> Variable:
    result = other ** self.value
    op = Exponentiation(other, self)
    return Variable(result, op)
