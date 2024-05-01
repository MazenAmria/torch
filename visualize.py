from autograd import Variable, log
from visual import visualize

a = Variable(1.0)
b = Variable(2.0)
c = Variable(3.0)
d = Variable(4.0)
e = Variable(5.0)
f = Variable(6.0)

Q = a + b * c + d ** e + log(f, 10)
Q.backward()

visualize(Q, 'graph.png')
