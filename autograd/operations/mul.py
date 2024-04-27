from ..operator import Operator
from ..variable import Variable


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
            other: Variable) -> Variable:
    result = self.value * other.value
    op = Multiply(self, other)
    return Variable(result, op)
