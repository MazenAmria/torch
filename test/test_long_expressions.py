from autograd import Variable


def test_long_expression_return_correct_result_and_gradient():
    a = Variable(4.0)
    b = Variable(-5.0)
    c = Variable(3.0)
    d = Variable(-6.0)

    Q = a**2 + b**5 + c*d
    Q.backward()

    assert Q.value == -3127
    assert a.grad == 8.0
    assert b.grad == 3125
    assert c.grad == -6.0
    assert d.grad == 3.0
