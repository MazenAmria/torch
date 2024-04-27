from autograd import Variable


def test_long_expression_return_correct_result_and_gradient():
    a = Variable(4.0)
    b = Variable(-5.0)
    c = Variable(3.0)
    d = Variable(-6.0)
    e = Variable(-3.0)
    f = Variable(4.0)

    Q = a**2 + b**5 + 3.0 * c * d + 7.0 + 3.0 * -e - f
    Q.backward()

    assert Q.value == -3151
    assert a.grad == 8.0
    assert b.grad == 3125
    assert c.grad == -18.0
    assert d.grad == 9.0
    assert e.grad == -3.0
    assert f.grad == -1.0
