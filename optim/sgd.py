from typing import Iterator
from numbers import Real

from autograd import Variable


class SGD:
    def __init__(self,
                 params: Iterator[Variable],
                 lr: Real = 1e-3) -> None:
        self.params = list(params)
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.value -= (param.grad * self.lr)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0.0
