#!/home/jupyter/py-env/python2.7.13/bin/python2.7

import theano
import theano.tensor as T

x = theano.tensor.vector() # declare variable
s = 1 / (1 + T.exp(-x))
f = theano.function([x], [s,s])   # compile function
print(f([0, 1, 2,0,0,0,1,1,1,0,-1,-3]))
