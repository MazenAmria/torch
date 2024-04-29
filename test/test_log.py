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
