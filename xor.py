from typing import List
from numbers import Real

from nn import Module
from nn.functional import sigmoid
from autograd import Variable, rand
from optim import SGD


class XorSolver(Module):
    def __init__(self) -> None:
        self.w_11 = rand()
        self.w_12 = rand()
        self.b_1 = rand()
        self.w_21 = rand()
        self.w_22 = rand()
        self.b_2 = rand()
        self.w_31 = rand()
        self.w_32 = rand()
        self.b_3 = rand()

    def forward(self, x: List[Real]) -> Variable:
        s_1 = x[0] * self.w_11 + x[1] * self.w_12 - self.b_1
        z_1 = sigmoid(s_1)
        
        s_2 = x[0] * self.w_21 + x[1] * self.w_22 - self.b_2
        z_2 = sigmoid(s_2)

        s_3 = z_1 * self.w_31 + z_2 * self.w_32 - self.b_3
        return sigmoid(s_3)

data = [
        [[0.0, 0.0], 0.0],
        [[0.0, 1.0], 1.0],
        [[1.0, 0.0], 1.0],
        [[1.0, 1.0], 0.0]
        ]
model = XorSolver()

optimizer = SGD(model.parameters(), lr=0.3)

for i in range(3000):
    optimizer.zero_grad()
    for X, y in data:
        y_p = model(X)
        loss = (y_p - y) ** 2
        loss.backward()
    optimizer.step()

for X, y in data:
    y_p = model(X)
    print(f'X = {X}, y_p = {y_p.value:.3f}, y = {y}')
