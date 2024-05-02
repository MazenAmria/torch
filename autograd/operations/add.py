from typing import Union
from numbers import Real

from ..operator import BinaryOperator
from ..variable import Variable


class Addition(BinaryOperator):
    def __init__(self,
                 a: Union[Variable, Real],
                 b: Union[Variable, Real]) -> None:
        super().__init__(a, b)

    def backward(self,
                 grad: Real = 1.0) -> None:
        if isinstance(self.a, Variable):
            self.a.backward(grad)
        if isinstance(self.b, Variable):
            self.b.backward(grad)


def variable_add(self: Variable, other: Union[Variable, Real]) -> Variable:
    if isinstance(other, Real):
        result = self.value + other
    elif isinstance(other, Variable):
        result = self.value + other.value
    else:
        raise TypeError(f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")
        
    op = Addition(self, other)
    return Variable(result, op)
