from typing import Union

from ..operator import Operator
from ..variable import Variable


class Scale(Operator):
    def __init__(self,
                 x: Variable,
                 c: float) -> None:
        self.x = x
        self.c = c

    def backward(self,
                 grad: float) -> None:
        self.x.backward(grad * self.c)


class Multiply(Operator):
    def __init__(self,
                 a: Variable,
                 b: Variable) -> None:
        self.a = a
        self.b = b

    def backward(self,
                 grad: float) -> None:
        self.a.backward(grad * self.b.value)
        self.b.backward(grad * self.a.value)


def variable_mul(self,
            other: Union[Variable, float]) -> Variable:
    if isinstance(other, float):
        result = self.value * other
        op = Scale(self, other)
        return Variable(result, op)
    elif isinstance(other, Variable):
        result = self.value * other.value
        op = Multiply(self, other)
        return Variable(result, op)
    else:
        raise TypeError(f"unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
