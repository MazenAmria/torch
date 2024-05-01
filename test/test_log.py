import numpy as np

from autograd import Variable, log
from .util import almost_equal


def test_log_would_return_correct_result():
    x = Variable(2.0)

    y = log(x)

    assert almost_equal(y.value, np.log(2.0))


def test_log_backward_would_return_corect_gradient():
    x = Variable(2.0)

    y = log(x)
    y.backward()

    assert x.grad == 0.5


def test_log_variable_base_would_return_correct_result():
    x = Variable(4.0)
    b = Variable(2.0)

    y = log(x, b)

    assert almost_equal(y.value, 2.0)


def test_log_variable_base_backward_would_return_corect_gradient():
    x = Variable(4.0)
    b = Variable(2.0)

    y = log(x, b)
    y.backward()

    assert almost_equal(x.grad, 0.25 * 1 / np.log(2.0))
    assert almost_equal(b.grad, np.log(4.0) * -0.5 / (np.log(2.0) ** 2))
