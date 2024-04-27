from typing import List

from nn import Module
from nn.functional import sigmoid
from autograd import Variable
from optim import SGD


class AndSolver(Module):
    def __init__(self) -> None:
        self.w_1 = Variable(0.0)
        self.w_2 = Variable(0.0)
        self.b = Variable(0.0)

    def forward(self, x: List[float]) -> Variable:
        s = x[0] * self.w_1 + x[1] * self.w_2 - self.b
        return sigmoid(s)

    def params(self) -> List[Variable]:
        return [self.w_1, self.w_2, self.b]


data = [
        [[0.0, 0.0], 0.0],
        [[0.0, 1.0], 0.0],
        [[1.0, 0.0], 0.0],
        [[1.0, 1.0], 1.0]
        ]
model = AndSolver()

optimizer = SGD(model.params(), learning_rate = 0.3)

for i in range(1000):
    optimizer.zero_grad()
    for X, y in data:
        y_p = model(X)
        loss = (y_p - y) ** 2
        loss.backward()
    optimizer.step()

for X, y in data:
    y_p = model(X)
    print(f'X = {X}, y_p = {y_p.value}, y = {y}')
