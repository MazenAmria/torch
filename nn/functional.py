from autograd import Variable, exp, zero


def sigmoid(x: Variable) -> Variable:
    return 1.0 / (1.0 + exp(-x))


def relu(x: Variable) -> Variable:
    return x if x.value >= 0.0 else zero()
