from .node import Node
from .operator import Operator


class Variable(Node):
    def __init__(self,
                 value: float,
                 parent: Node = None) -> None:
        self.value = value
        self.parent = parent
        self.grad = 0.0

    def backward(self,
                 grad: float = 1.0) -> None:
        self.grad += grad
        if self.parent is not None:
            self.parent.backward(grad)
