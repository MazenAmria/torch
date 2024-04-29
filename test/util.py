
import numpy as np


def almost_equal(a: float, b: float, precision: float = 1e-5):
    return abs(a - b) < precision
