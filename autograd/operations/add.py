from typing import Union, List

from ..operator import BinaryOperator
from ..variable import Variable


class Add(BinaryOperator):
    def __init__(self,
                 a: Variable,
                 b: Variable) -> None:
        super().__init__(a, b)

    def backward(self,
                 grad: float) -> None:
        self.a.backward(grad)
        self.b.backward(grad)


def variable_add(self: Variable, other: Union[Variable, float]) -> Variable:
    if isinstance(other, float):
        result = self.value + other
        return Variable(result, self)
    elif isinstance(other, Variable):
        result = self.value + other.value
        op = Add(self, other)
        return Variable(result, op)
    else:
        raise TypeError(f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")
