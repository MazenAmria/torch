from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def backward(self,
                 grad: float) -> None:
        pass


class Operator(Node):
    pass


class Variable(Node):
    def __init__(self,
                 value: float,
                 parent: Operator = None) -> None:
        self.value = value
        self.parent = parent
        self.grad = 0.0

    def backward(self,
                 grad: float = 1.0) -> None:
        self.grad += grad
        if self.parent is not None:
            self.parent.backward(grad)


class AdditionOperator(Operator):
    def __init__(self,
                 a: Variable,
                 b: Variable) -> None:
        self.a = a
        self.b = b

    def backward(self,
                 grad: float) -> None:
        self.a.backward(grad)
        self.b.backward(grad)


class MultiplicationOperator(Operator):
    def __init__(self,
                 a: Variable,
                 b: Variable) -> None:
        self.a = a
        self.b = b

    def backward(self,
                 grad: float) -> None:
        self.a.backward(grad * self.b.value)
        self.b.backward(grad * self.a.value)


class ExponentiationOperator(Operator):
    def __init__(self,
                 x: Variable,
                 p: float) -> None:
        self.x = x
        self.p = p

    def backward(self,
                 grad: float) -> None:
        self.x.backward(grad * self.x.value ** (self.p - 1) * self.p)


def variable_add(self,
            other: Variable) -> Variable:
    result = self.value + other.value
    op = AdditionOperator(self, other)
    return Variable(result, op)


def variable_mul(self,
            other: Variable) -> Variable:
    result = self.value * other.value
    op = MultiplicationOperator(self, other)
    return Variable(result, op)


def variable_pow(self,
            p: float) -> Variable:
    result = self.value ** p
    op = ExponentiationOperator(self, p)
    return Variable(result, op)


Variable.__add__ = variable_add
Variable.__mul__ = variable_mul
Variable.__pow__ = variable_pow

a = Variable(2.0)
b = Variable(-3.0)
c = Variable(10.0)
d = Variable(-2.0)

Q = (a * b + c) * d
Q.backward()

print("Q = (a * b + c) * d")
print(f"a = {a.value}, b = {b.value}, c = {c.value}, d = {d.value}")
print(f"dQ/da = {a.grad}")
print(f"dQ/db = {b.grad}")
print(f"dQ/dc = {c.grad}")
print(f"dQ/dd = {d.grad}")

x = Variable(4.0)
Q = x ** 5
Q.backward()

print("Q = x^5")
print(f"x = {x.value}")
print(f"dQ/dx = {x.grad}")
