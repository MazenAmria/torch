from autograd import Variable


def test_add_would_return_correct_sum():
    a = Variable(4.0)
    b = Variable(-5.0)

    c = a + b

    assert c.value == -1.0


def test_add_backward_would_return_corect_gradients():
    a = Variable(4.0)
    b = Variable(-5.0)

    c = a + b
    c.backward()

    assert a.grad == 1.0
    assert b.grad == 1.0
