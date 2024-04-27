from .variable import Variable
from .operations.add import variable_add
from .operations.mul import variable_mul
from .operations.neg import variable_neg
from .operations.pow import variable_pow


Variable.__add__ = variable_add
Variable.__radd__ = variable_add
Variable.__neg__ = variable_neg
Variable.__sub__ = lambda a, b: a + -b
Variable.__rsub__ = lambda b, a: -b + a
Variable.__mul__ = variable_mul
Variable.__rmul__ = variable_mul
Variable.__truediv__ = lambda a, b: a * b**-1
Variable.__rtruediv__ = lambda b, a: b**-1 * a
Variable.__pow__ = variable_pow

__exports__ = { 'Variable': Variable }
