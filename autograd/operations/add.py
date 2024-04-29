from typing import Union, List
from numbers import Number

from ..operator import UnaryOperator, BinaryOperator
from ..variable import Variable


class AddConstant(UnaryOperator):
    def __init__(self,
                 x: Variable,
                 c: Number) -> None:
        super().__init__(x)

    def backward(self,
                 grad: Number) -> None:
        self.x.backward(grad)


class Add(BinaryOperator):
    def __init__(self,
                 a: Variable,
                 b: Variable) -> None:
        super().__init__(a, b)

    def backward(self,
                 grad: Number) -> None:
        self.a.backward(grad)
        self.b.backward(grad)


def variable_add(self: Variable, other: Union[Variable, Number]) -> Variable:
    if isinstance(other, Number):
        result = self.value + other
        op = AddConstant(self, other)
        return Variable(result, op)
    elif isinstance(other, Variable):
        result = self.value + other.value
        op = Add(self, other)
        return Variable(result, op)
    else:
        raise TypeError(f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")
