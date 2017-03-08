#!/home/jupyter/py-env/python2.7.13/bin/python2.7

import theano
import theano.tensor as T
from theano import pp

x  = T.dmatrix('x') # declare variable
y  = T.sum(1 / (1 + T.exp(-x)))
dy = T.grad(y,x)
f  = theano.function([x], dy)   # compile function
print f([[-1,0],[1,2]])

