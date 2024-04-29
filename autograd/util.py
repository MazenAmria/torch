from typing import Union
from numbers import Number

import numpy as np

from .variable import Variable
from .operations.log import variable_log


def exp(x: Union[Variable, Number]) -> Variable:
    return Variable(np.e) ** x


def log(x: Union[Variable, Number], b: Number = None) -> Variable:
    if isinstance(x, Number):
        x = Variable(x)
    elif not isinstance(x, Variable):
        raise TypeError(f'cannot use log for type {type(x)}, only Variable and Number types are allowed.')

    return variable_log(x, b)
