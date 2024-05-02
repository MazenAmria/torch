from numbers import Real

from .node import Node


class Variable(Node):
    def __init__(self,
                 value: Real,
                 parent: Node = None,
                 name: str = None) -> None:
        self.value = value
        self.parent = parent
        self.name = name
        self.grad = 0.0

    def backward(self,
                 grad: Real = 1.0) -> None:
        self.grad += grad
        if self.parent is not None:
            self.parent.backward(grad)
