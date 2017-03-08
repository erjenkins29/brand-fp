#!/home/jupyter/py-env/python2.7.13/bin/python2.7

import theano
import theano.tensor as T
from theano import pp

# needs to be double type, so derivatives can occur
x  = T.dvector('x') # declare variable
y  = 1 / (1 + T.exp(-x))

# use scan function to compute the gradient at y[i] wrt input x.  
# NOTE: not sure what updates does... probably an accumulator?
# NOTE: The order of the lambda variables is not trivial.  i corresponds to the the sequences param.
J, updates = theano.scan(lambda i,y,x : T.grad(y[i],x),sequences=T.arange(y.shape[0]),non_sequences=[y,x])

f  = theano.function([x], J, updates=updates)   # compile function
print f([-1,0,3,4])

