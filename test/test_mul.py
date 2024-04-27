from autograd import Variable


def test_mul_would_return_the_correct_product():
    a = Variable(4.0)
    b = Variable(-5.0)

    c = a * b

    assert c.value == -20.0


def test_mul_backward_would_return_corect_gradients():
    a = Variable(4.0)
    b = Variable(-5.0)

    c = a * b
    c.backward()

    assert a.grad == -5.0
    assert b.grad == 4.0


def test_mul_by_constant_return_correct_sum():
    x = Variable(4.0)

    y = 3.0 * x

    assert y.value == 12.0


def test_mul_by_constant_return_correct_gradient():
    x = Variable(4.0)

    y = 3.0 * x
    y.backward()

    assert x.grad == 3.0
