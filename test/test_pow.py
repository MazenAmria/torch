import numpy as np

from autograd import Variable, exp
from .util import almost_equal


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

    y = exp(x)
    
    assert almost_equal(y.value, np.exp(4.0))


def test_exp_backward_would_return_correct_gradient():
    x = Variable(4.0)

    y = exp(x)
    y.backward()
    
    assert almost_equal(x.grad, np.exp(4.0))


def test_var_exp_would_return_correct_result():
    a = Variable(np.e)
    b = Variable(4.0)

    c = a ** b
    
    assert almost_equal(c.value, np.exp(4.0))


def test_var_exp_backward_would_return_correct_gradients():
    a = Variable(np.e)
    b = Variable(4.0)

    c = a ** b
    c.backward()
    
    assert almost_equal(a.grad, 4 * np.exp(3.0))
    assert almost_equal(b.grad, np.exp(4.0))
