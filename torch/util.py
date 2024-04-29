from typing import Union
from numbers import Number

import numpy as np

from autograd import Variable


def exp(x: Union[Variable, Number]) -> Variable:
    return Variable(np.e) ** x
