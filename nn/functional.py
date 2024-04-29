from autograd import Variable, exp


def sigmoid(x: Variable) -> Variable:
    return 1.0 / (1.0 + exp(-x))
