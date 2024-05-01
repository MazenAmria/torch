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


def log(x: Union[Variable, Number], b: Union[Variable, Number] = None) -> Union[Variable, Number]:
    denominator = log(b) if b is not None else None

    if isinstance(x, Number):
        result = np.log(x)
    if isinstance(x, Variable):
        result = variable_log(x)
    else:
        raise TypeError(f"unsupported argument type for log: '{type(x)}'")

    if denominator is not None:
        return result / denominator
    else:
        return result
