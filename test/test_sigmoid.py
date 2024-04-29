from nn.functional import sigmoid
from autograd import Variable
from .util import almost_equal


def test_sigmoid_has_correct_gradient():
    x = Variable(0.0)
    y = sigmoid(x)
    y.backward()

    assert almost_equal(y.value, 0.5)
    assert almost_equal(x.grad, 0.25)
