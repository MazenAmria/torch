from .variable import Variable
from .operations.add import variable_add
from .operations.mul import variable_mul
from .operations.pow import variable_pow


Variable.__add__ = variable_add
Variable.__mul__ = variable_mul
Variable.__pow__ = variable_pow

__exports__ = { 'Variable': Variable }
