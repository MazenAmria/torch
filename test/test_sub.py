from autograd import Variable


def test_sub_would_return_correct_result():
    a = Variable(4.0)
    b = Variable(-5.0)

    c = a - b

    assert c.value == 9.0


def test_sub_backward_would_return_corect_gradients():
    a = Variable(4.0)
    b = Variable(-5.0)

    c = a - b
    c.backward()

    assert a.grad == 1.0
    assert b.grad == -1.0


def test_sub_to_constant_return_correct_sum():
    x = Variable(4.0)

    y = x - 3.0
    
    assert y.value == 1.0
    
    y = 3.0 - x

    assert y.value == -1.0


def test_sub_to_constant_return_correct_gradient():
    x = Variable(4.0)

    y = x - 3.0
    y.backward()

    assert x.grad == 1.0

    x.grad = 0.0
    
    y = 3.0 - x
    y.backward()

    assert x.grad == -1.0
