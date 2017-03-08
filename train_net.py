#!/home/jupyter/py-env/py2.7.13/bin/python2.7

import theano
from theano import function, pp
from theano import tensor as T
import numpy as np

## Define sigmoid layer
def sigmoid_layer(x,w):
   """
   INPUT - 
   x:  input vector  (numpy array)
   w:  weight matrix (numpy 2darray)

   OUTPUT -  element-wise sigmoid of...
   [x (with an added bias term) times the weight matrix w]
   """
   b = np.array([1], dtype=theano.config.floatX)
   x_with_bias = T.concatenate([x,b]) # attach the bias term to end of vector
   z = T.dot(w.T, x_with_bias)
   y = T.nnet.sigmoid(z)
   return y

## Define the output layer

########### function is not clear right now.
#### options:
####   - log linear
####   - linear
####   - bins (log)

## Define gradient descent function
def grad_desc(cost, theta):
   """
   cost:  function which we are minimizing
   theta: input vector
   """
   alpha = .1     #learning rate
   return theta - (alpha * T.grad(cost,wrt=theta))

## Declare variables

## input vector
x = T.dvector('x')

## output vector (or value?)
y = T.dscalar('y')

## weight matrix -- 
##     input features = 30, outputs to a 8-node hidden layer
theta1 = theano.shared(np.array(np.random.rand(30,8), dtype=theano.config.floatX))
theta2 = theano.shared(np.array(np.random.rand(8,1), dtype=theano.config.floatX))


### hidden layer
h = sigmoid_layer(x, theta1)

### output layer
o = T.sum(T.dot(h.T,theta2))

## State cost function
cost = (o - y)**2
f_cost = theano.function([x,y], cost, updates=[
     (theta1, grad_desc(cost, theta1)),
     (theta2, grad_desc(cost, theta2))])
f_overall = theano.function([x],o)

print "got through the script without error..."

##### TODO #####
# Read in the the training data here.
## Output trained weights
### print costs as iterate through training epochs
###############


