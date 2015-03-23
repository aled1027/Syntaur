"""
layers.py
"""

import theano
import theano.tensor as T
import numpy as np

# GLOBAL
PRNG = np.random
SIGMOID = T.nnet.sigmoid
TANH = T.tanh
SOFTMAX = T.nnet.softmax

# HELPERS
def INIT_LO(dim_in, dim_out): return -1*np.sqrt( 6. / (dim_in + dim_out) )
def INIT_HI(dim_in, dim_out): return np.sqrt( 6. / (dim_in + dim_out) )

class Layer(object):
    def __init__(self, dim_in, dim_out, prng = PRNG, activation = SIGMOID):
        self.CONNECTED = False
        self.activation = activation

        # initialize parameters
        self.W = theano.shared(
            np.asarray(
                prng.uniform(
                    low = INIT_LO(dim_in, dim_out),
                    high = INIT_HI(dim_in, dim_out),
                    size = (dim_in, dim_out)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        self.b = theano.shared(
            np.zeros((dim_out), dtype = theano.config.floatX),
            borrow = True
        )

        self.params = [self.W, self.b]
        
    def connect(self, X, v):
        self.X = X
        self.v = v
        self.net_in = T.dot(self.X, self.W) + self.b
        self.vec_in = T.dot(self.v, self.W) + self.b
        if self.activation is None:
            self.output = self.net_in
            self.vec_out = self.vec_in

        else:
            self.output = self.activation(self.net_in)
            self.vec_out = self.activation(self.vec_in)
        self.prediction = T.argmax(self.vec_out)
        self.predicter = theano.function([self.v], self.prediction)
        self.CONNECTED = True

    def predict(self, x):
        if not self.CONNECTED:
            raise RuntimeError("Asked to predict, but I'm not yet connected.")
        return self.predicter(x)


class OutputLayer(Layer):
    def __init__(self, dim_in, dim_out, activation = None):
        super(OutputLayer, self).__init__(dim_in, dim_out, activation = activation)
        self.W.set_value(np.zeros((dim_in, dim_out), dtype = theano.config.floatX))

    def connect(self, X, v):
        super(OutputLayer, self).connect(X, v)

    def cost(self, y):
        raise NotImplementedError("Output Layer must have a cost function.")

    def errors(self, y):
        raise NotImplementedError("Output Layer must have an error function.")


class SoftmaxLayer(OutputLayer):
    def __init__(self, dim_in, dim_out):
        super(SoftmaxLayer, self).__init__(dim_in, dim_out, activation = SOFTMAX)


    def connect(self, X, v):
        super(SoftmaxLayer, self).connect(X, v)
        self.p_y_given_x = self.output
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

    def negative_log_likelihood(self, y):
        if not self.CONNECTED:
            raise RuntimeError("Asked to compute a negative log likehilhood but I'm not connected atm")
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def cost(self, y):
        if not self.CONNECTED:
            raise RuntimeError("Asked to compute a cost function, but I'm not connected atm")
        return self.negative_log_likelihood(y)

    def errors(self, y, mean = False):
        if not self.CONNECTED:
            raise RuntimeError("Asked to compute errors, but I'm not connected atm")
        if mean:
            return T.mean(T.neq(self.y_pred, y))
        else:
            return T.neq(self.y_pred, y)

