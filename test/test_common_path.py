from autograd import Variable


def test_common_path_return_correct_result_and_gradient():
    a = Variable(4.0)
    b = 4 * a
    c = 3 * b
    d = 7 * b
    Q = c + d
    Q.backward()

    assert Q.value == 160.0
    assert a.grad == 40.0
    assert b.grad == 10.0
    assert c.grad == 1.0
    assert d.grad == 1.0
