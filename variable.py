class AdditionOperator:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad):
        self.a.backward(grad)
        self.b.backward(grad)

class MultiplicationOperator:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad):
        self.a.backward(grad * self.b.value)
        self.b.backward(grad * self.a.value)

class ExponentiationOperator:
    def __init__(self, x, p):
        self.x = x
        self.p = p

    def backward(self, grad):
        self.x.backward(grad * self.x.value ** (self.p - 1) * self.p)

class Variable:
    def __init__(self, value, parent = None):
        self.value = value
        self.parent = parent
        self.grad = 0.0

    def __add__(self, other):
        result = self.value + other.value
        op = AdditionOperator(self, other)
        return Variable(result, op)

    def __mul__(self, other):
        result = self.value * other.value
        op = MultiplicationOperator(self, other)
        return Variable(result, op)

    def __pow__(self, p):
        result = self.value ** p
        op = ExponentiationOperator(self, p)
        return Variable(result, op)

    def backward(self, grad=1.0):
        self.grad += grad
        if self.parent is not None:
            self.parent.backward(grad)


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
