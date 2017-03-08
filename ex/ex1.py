#!/home/jupyter/py-env/python2.7.13/bin/python2.7

import theano
a = theano.tensor.vector() # declare variable
out = a + a                # build symbolic expression
f = theano.function([a], out)   # compile function
print(f([0, 1, 2]))
