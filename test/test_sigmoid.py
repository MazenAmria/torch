from nn.functional import sigmoid
from autograd import Variable


def test_sigmoid_has_correct_gradient():
    x = Variable(0.0)
    y = sigmoid(x)
    y.backward()

    assert abs(y.value - 0.5) < 1e-3
    assert abs(x.grad - 0.25) < 1e-3
