from autograd import Variable


def test_div_would_return_correct_result():
    a = Variable(4.0)
    b = Variable(-2.0)

    c = a / b

    assert c.value == -2.0


def test_div_backward_would_return_corect_gradients():
    a = Variable(4.0)
    b = Variable(-2.0)

    c = a / b
    c.backward()

    assert a.grad == -0.5
    assert b.grad == -1.0


def test_div_with_constant_return_correct_result():
    x = Variable(4.0)

    y = x / 2.0

    assert y.value == 2.0
    
    y = 8.0 / x

    assert y.value == 2.0


def test_div_by_constant_return_correct_gradient():
    x = Variable(4.0)

    y = x / 2.0
    y.backward()

    assert x.grad == 0.5

    x.grad = 0.0

    y = 8.0 / x
    y.backward()

    assert x.grad == -0.5
