from typing import Union
from numbers import Number

import numpy as np

from .variable import Variable
from .operations.log import variable_log


def exp(x: Union[Variable, Number]) -> Union[Variable, Number]:
    if isinstance(x, Variable):
        return np.e ** x
    if isinstance(x, Number):
        return np.exp(x)
    else:
        raise TypeError(f"unsupported argument type for exp: '{type(x)}'")

def log(x: Union[Variable, Number], b: Number = None) -> Union[Variable, Number]:
    if isinstance(x, Number):
        if b is not None: 
            return np.log(x) / np.log(b)
        else:
            return np.log(x)
    if isinstance(x, Variable):
        return variable_log(x, b)
    else:
        raise TypeError(f"unsupported argument type for log: '{type(x)}'")
