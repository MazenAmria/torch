from ..operator import Operator
from ..variable import Variable


class Power(Operator):
    def __init__(self,
                 x: Variable,
                 p: float) -> None:
        self.x = x
        self.p = p

    def backward(self,
                 grad: float) -> None:
        self.x.backward(grad * self.x.value ** (self.p - 1) * self.p)


def variable_pow(self,
            p: float) -> Variable:
    result = self.value ** p
    op = Power(self, p)
    return Variable(result, op)
