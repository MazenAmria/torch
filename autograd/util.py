from typing import Union
from numbers import Real

import numpy as np

from .variable import Variable
from .operations.log import variable_log


def exp(x: Union[Variable, Real]) -> Union[Variable, Real]:
    if isinstance(x, Variable):
        return np.e ** x
    if isinstance(x, Real):
        return np.exp(x)
    else:
        raise TypeError(f"unsupported argument type for exp: '{type(x)}'")


def log(x: Union[Variable, Real], b: Union[Variable, Real] = None) -> Union[Variable, Real]:
    denominator = log(b) if b is not None else None

    if isinstance(x, Real):
        result = np.log(x)
    if isinstance(x, Variable):
        result = variable_log(x)
    else:
        raise TypeError(f"unsupported argument type for log: '{type(x)}'")

    if denominator is not None:
        return result / denominator
    else:
        return result


def rand() -> Variable:
    return Variable(np.random.random_sample())


def zero() -> Variable:
    return Variable(0.0)
