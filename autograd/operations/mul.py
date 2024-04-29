from typing import Union
from numbers import Number

from ..operator import UnaryOperator, BinaryOperator
from ..variable import Variable


class Scale(UnaryOperator):
    def __init__(self,
                 x: Variable,
                 c: Number) -> None:
        super().__init__(x)
        self.c = c

    def backward(self,
                 grad: Number) -> None:
        self.x.backward(grad * self.c)


class Multiply(BinaryOperator):
    def __init__(self,
                 a: Variable,
                 b: Variable) -> None:
        super().__init__(a, b)

    def backward(self,
                 grad: Number) -> None:
        self.a.backward(grad * self.b.value)
        self.b.backward(grad * self.a.value)


def variable_mul(self: Variable,
            other: Union[Variable, Number]) -> Variable:
    if isinstance(other, Number):
        result = self.value * other
        op = Scale(self, other)
        return Variable(result, op)
    elif isinstance(other, Variable):
        result = self.value * other.value
        op = Multiply(self, other)
        return Variable(result, op)
    else:
        raise TypeError(f"unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
