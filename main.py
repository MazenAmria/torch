from autograd import Variable

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
