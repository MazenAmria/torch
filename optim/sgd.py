from typing import List

from autograd import Variable


class SGD:
    def __init__(self,
                 params: List[Variable],
                 lr: float = 1e-3) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.value -= (param.grad * self.lr)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0.0
