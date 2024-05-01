from typing import Union, List
from numbers import Number

from ..operator import BinaryOperator
from ..variable import Variable


class Add(BinaryOperator):
    def __init__(self,
                 a: Union[Variable, Number],
                 b: Union[Variable, Number]) -> None:
        super().__init__(a, b)

    def backward(self,
                 grad: Number = 1.0) -> None:
        if isinstance(self.a, Variable):
            self.a.backward(grad)
        if isinstance(self.b, Variable):
            self.b.backward(grad)


def variable_add(self: Variable, other: Union[Variable, Number]) -> Variable:
    if isinstance(other, Number):
        result = self.value + other
    elif isinstance(other, Variable):
        result = self.value + other.value
    else:
        raise TypeError(f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")
        
    op = Add(self, other)
    return Variable(result, op)
