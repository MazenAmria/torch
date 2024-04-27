import numpy as np

from autograd import Variable


def almost_equal(a: float, b: float, precision: float = 1e-5):
    return abs(a - b) < precision


def test_pow_would_return_correct_result():
    x = Variable(2.0)

    y = x ** 4

    assert y.value == 16.0


def test_pow_backward_would_return_corect_gradient():
    x = Variable(2.0)

    y = x ** 4
    y.backward()

    assert x.grad == 32.0


def test_exp_would_return_correct_result():
    x = Variable(4.0)

    y = np.e ** x
    
    assert almost_equal(y.value, np.exp(4.0))


def test_exp_backward_would_return_correct_gradient():
    x = Variable(4.0)

    y = np.e ** x
    y.backward()
    
    assert almost_equal(x.grad, np.exp(4.0))
