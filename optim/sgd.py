from typing import List

from autograd import Variable


class SGD:
    def __init__(self,
                 params: List[Variable],
                 learning_rate: float = 1e-3) -> None:
        self.params = params
        self.learning_rate = learning_rate

    def step(self) -> None:
        for param in self.params:
            param.value -= (param.grad * self.learning_rate)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0.0
