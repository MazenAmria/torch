import numpy as np

from autograd import Variable

def sigmoid(x: Variable) -> Variable:
    return 1.0 / (1.0 + np.e ** -x)
