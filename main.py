from typing import List

from nn import Module
from nn.functional import sigmoid
from autograd import Variable


class AndSolver(Module):
    def __init__(self) -> None:
        self.w1 = Variable(0.0)
        self.w2 = Variable(0.0)
        self.b = Variable(0.0)

    def forward(self, x: List[float]) -> Variable:
        s = x[0] * self.w1 + x[1] * self.w2 + self.b
        return sigmoid(s)

    def params(self) -> List[Variable]:
        return [self.w1, self.w2, self.b]


data = [
        [[0.0, 0.0], 0.0],
        [[0.0, 1.0], 0.0],
        [[0.0, 0.0], 0.0],
        [[1.0, 1.0], 1.0]
        ]
model = AndSolver()

for X, y in data:
    out = model(X)
    print(f'predicted = {out.value}, actual = {y}')
