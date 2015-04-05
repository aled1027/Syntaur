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

def AS_ONEHOT(v, n_dims, as_int = True):
    if as_int:
        return T.extra_ops.to_one_hot(v, nb_class = n_dims, dtype = 'int32')
    else:
        return T.extra_ops.to_one_hot(v, nb_class = n_dims, 
                                      dtype = theano.config.floatX)

class Layer(object):
    def __init__(self, dim_in, dim_out, prng = PRNG, activation = SIGMOID):
        self.CONNECTED = False
        self.activation = activation
        self.dim_in = dim_in
        self.dim_out = dim_out

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

    def __str__(self):
        return "<Generic Layer> with %d inputs and %d outputs" %(self.dim_in, self.dim_out)

        
    def connect(self, X):
        self.X = X
        self.net_in = T.dot(self.X, self.W) + self.b

        if self.activation is None:
            self.output = self.net_in


        else:
            self.output = self.activation(self.net_in)

        self.prediction = T.argmax(self.output, axis=1)
        self.predicter = theano.function([self.X], self.prediction)
        self.CONNECTED = True

    def predict(self, x):
        if not self.CONNECTED:
            raise RuntimeError("Asked to predict, but I'm not yet connected.")
        return self.predicter(x)

class RecurrentLayer(object):
    def __init__(self, dim_in, dim_hidden, dim_out, prng = PRNG, activation = SIGMOID):
        self.CONNECTED = False
        self.activation = activation
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        # Initial state of recurrent layer.
        self.h_init = theano.shared(
            np.asarray(
                prng.uniform(
                    low = INIT_LO(dim_hidden, dim_hidden),
                    high =INIT_HI(dim_hidden, dim_hidden),
                    size = (dim_hidden,)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        self.W_ih = theano.shared(
            np.asarray(
                prng.uniform(
                    low = INIT_LO(dim_in, dim_hidden),
                    high = INIT_HI(dim_in, dim_hidden),
                    size = (dim_in, dim_hidden)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        self.W_hh = theano.shared(
            np.asarray(
                prng.uniform(
                    low = INIT_LO(dim_hidden, dim_hidden),
                    high = INIT_HI(dim_hidden, dim_hidden),
                    size = (dim_hidden, dim_hidden)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        self.W_ho = theano.shared(
            np.asarray(
                prng.uniform(
                    low = INIT_LO(dim_hidden, dim_out),
                    high = INIT_HI(dim_hidden, dim_out),
                    size = (dim_hidden, dim_out)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )
                
        self.params = [self.h_init, self.W_ih, self.W_hh, self.W_ho]
        
    def set_h_init(self, v):
        self.h_init = v
        if self.CONNECTED:
            self.connect(self, self.S) # re-connect so that new h_init is used.
        self.params = [self.h_init, self.W_ih, self.W_hh, self.W_ho]

    def connect(self, S):
        self.S = S
        def step(s_current, h_prev):
            h_t = self.activation(
                T.dot(s_current, self.W_ih) + 
                T.dot(h_prev, self.W_hh)
            )
            y_t = self.activation(
                T.dot(h_t, self.W_ho)
            )
            return h_t, y_t
        [self.H, self.output], _ = theano.scan(
            step,
            sequences = self.S,
            outputs_info = [self.h_init, None]
        )
        
        self.prediction, _ = theano.map(
            lambda x: T.argmax(x),
            sequences = self.output
        )
        self.final_state = self.H[self.H.shape[0] - 1]
        self.outputter = theano.function([self.S], self.output)
        self.predicter = theano.function([self.S], self.prediction)
        self.CONNECTED = True
        
    def predict(self, X):
        if not self.CONNECTED:
            raise RuntimeError("Asked to predict, but I'm not yet connected.")
        return self.predicter(X)
        

class OutputLayer(Layer):
    def __init__(self, dim_in, dim_out, activation = None):
        super(OutputLayer, self).__init__(dim_in, dim_out, activation = activation)
        self.W.set_value(np.zeros((dim_in, dim_out), dtype = theano.config.floatX))

    def __str__(self):
        activ_str = "Linear"
        if self.activation == SIGMOID:
            activ_str = "Sigmoid"
        if self.activation == SOFTMAX:
            activ_str = "Softmax"
        if self.activation == TANH:
            activ_str = "Tanh"
        return "<Output Layer> with %d inputs, %d outputs, %s activation" %(self.dim_in, self.dim_out, activ_str)

    def connect(self, X):
        super(OutputLayer, self).connect(X)

    def cost(self, y):
        raise NotImplementedError("Output Layer must have a cost function.")

    def errors(self, y):
        raise NotImplementedError("Output Layer must have an error function.")


class SoftmaxLayer(OutputLayer):
    def __init__(self, dim_in, dim_out):
        super(SoftmaxLayer, self).__init__(dim_in, dim_out, activation = SOFTMAX)

    def connect(self, X):
        super(SoftmaxLayer, self).connect(X)
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

class ConnectionSpec(object):
    """
    Passed as input to a NN constructor. Contains a list of layers and
    connections between layers. Provides a function for connecting up layers once the inputs are provided. 
    """
    def __init__(self, layers, connections):
        """
        
        :type layers: list
        :param layers: A list of 'Layer' objects (unconnected).
        
        :type connections: dict
        :param connections: a dictionary of connections whose keys are source layer indices 
        and whose values are destination layer indices. 
        """
        self.layers = layers
        self.input_layers = []
        self.output_layers = []
        self.CONNECTED = False

        # Determine which layers are receiving (non-input)
        receiver_idxs = set(sum(connections.values(),[]))

        # Determine input, output layers
        for (i,l) in enumerate(layers):
            if i not in connections.keys() and i not in receiver_idxs:
                raise RuntimeError("Layer %s not used in computation" %str(l))
            elif i in connections.keys() and i not in receiver_idxs:
                self.input_layers.append(l)
            elif i not in connections.keys() and i in receiver_idxs:
                self.output_layers.append(l)

        # get connect func
        def connect_func():
            if self.CONNECTED:
                raise RuntimeError("Connect_func has already been run.")
            for src_idx in connections:
                for dest_idx in connections[src_idx]:
                    src, dest = (layers[src_idx], layers[dest_idx])
                    if not src.CONNECTED:
                        raise RuntimeError("This input layer isn't connected.")
                    dest.connect(src.output)
            self.CONNECTED = True
            
        self.connecter = connect_func







