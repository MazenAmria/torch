import numpy as np

from autograd import Variable

def sigmoid(x: Variable) -> Variable:
    return (np.e ** x) / (np.e ** x + 1.0)
