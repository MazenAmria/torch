from autograd import Variable


def test_neg_would_return_correct_value():
    x = Variable(4.0)

    y = -x

    assert y.value == -4.0


def test_neg_backward_would_return_corect_gradient():
    x = Variable(4.0)

    y = -x
    y.backward()

    assert x.grad == -1.0

