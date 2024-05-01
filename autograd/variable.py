from typing import List
from numbers import Number

from .node import Node


class Variable(Node):
    def __init__(self,
                 value: Number,
                 parent: Node = None) -> None:
        self.value = value
        self.parent = parent
        self.grad = 0.0

    def backward(self,
                 grad: Number = 1.0) -> None:
        self.grad += grad
        if self.parent is not None:
            self.parent.backward(grad)

    def parameters(self) -> List['Variable']:
        if self.parent is None:
            return [self]
        return self.parent.parameters()
