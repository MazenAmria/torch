from autograd import Variable


def test_pow_would_return_correct_exponential():
    x = Variable(2.0)

    y = x ** 4

    assert y.value == 16.0


def test_add_backward_would_return_corect_gradients():
    x = Variable(2.0)

    y = x ** 4
    y.backward()

    assert x.grad == 32.0
