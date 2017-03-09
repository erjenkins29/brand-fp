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
theta1 = theano.shared(np.array(np.random.rand(301,8), dtype=theano.config.floatX))
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

#print "got through the script without error..."

# Read in the training data here.
from numpy import genfromtxt

#### Manual: choose 1,2,3, or 4 here ####
of_interest = 1
#########################################
targets = {1: 'qty_amt', 2:'rev_amt', 3:'qty_share', 4:'rev_share'}

print "generating model for when the target is", targets[of_interest]

data = genfromtxt('sampledata_%s.csv' % targets[of_interest],delimiter=',')

data    = data[1:,2:]  # cut the header and the first two rows (identifiers)
x_in    = data[:,:-2]  # rank bit-vectors
y_out   = data[:,-2]   # the target variable
#### TODO: involve the agg1 field at data[:,-1]

x_valid = x_in[-1,:]   # keep one row for predicting an example output
x_in    = x_in[:-1,:]  # remove validation row from input data


### Normalize the target variable - otherwise the cost grows too high
if of_interest<=2: 
   y_out = y_out / 20000.

print "\nBeginning model training.."
print "y_out:\n\n",y_out[1]
print "x_in: \n\n",x_in[1]
print "\n\ntraining size:",x_in.shape

cur_cost = 0
acceptable_cost = 1e-5

for i in range(10000):
   for k in range(len(x_in)):
      cur_cost = f_cost(x_in[k], y_out[k])
   if i % 50 == 0: print "Cost:",cur_cost 
   if cur_cost < acceptable_cost: 
      print "acceptable weight vectors found at iteration",i
      break

## Output trained weights
print theta1.get_value()
print theta2.get_value()
#print theta1

print "---- Validation ----\n\nexample input [x]:", x_valid
print "predicted %s = %i" %(targets[of_interest], int(20000*f_overall(x_valid)))
### print costs as iterate through training epochs
###############


