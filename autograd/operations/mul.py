from typing import Union
from numbers import Number

from ..operator import BinaryOperator
from ..variable import Variable


class Multiplication(BinaryOperator):
    def __init__(self,
                 a: Union[Variable, Number],
                 b: Union[Variable, Number]) -> None:
        super().__init__(a, b)

    def backward(self,
                 grad: Number = 1.0) -> None:
        a_val = self.a.value if isinstance(self.a, Variable) else self.a
        b_val = self.b.value if isinstance(self.b, Variable) else self.b

        if isinstance(self.a, Variable):
            self.a.backward(grad * b_val)
        if isinstance(self.b, Variable):
            self.b.backward(grad * a_val)


def variable_mul(self: Variable, other: Union[Variable, Number]) -> Variable:
    if isinstance(other, Number):
        result = self.value * other
    elif isinstance(other, Variable):
        result = self.value * other.value
    else:
        raise TypeError(f"unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
    
    op = Multiplication(self, other)
    return Variable(result, op)
