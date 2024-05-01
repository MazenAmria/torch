from autograd import Variable, log
from visual import visualize

a = Variable(1.0, name='a')
b = Variable(2.0, name='b')
c = Variable(3.0, name='c')
d = Variable(4.0, name='d')
e = Variable(5.0, name='e')
f = Variable(6.0, name='f')
g = Variable(10.0, name='g')

Q = a + b * c + d ** e + log(f, g)
Q.name = 'Q'
Q.backward()

visualize(Q, 'graph.png')
