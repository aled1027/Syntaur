"""
models.py
"""

from syntaur.layers import Layer, SoftmaxLayer, RecurrentLayer, ConnectionSpec
import text_util
import theano
import theano.tensor as T

class Model(object):
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in 
        self.dim_out = dim_out
        self.X = T.dmatrix()
        self.v = T.dvector()
        
    def predict(self, x):
        raise NotImplementedError("Model must provide a predict(.) method.")
    
    def cost(self, y):
        raise NotImplementedError("Model must have a cost(.) method.")

    def errors(self, y):
        raise NotImplementedError("Model must have a errors(.) method.")

class LogisticRegression(Model):
    def __init__(self, dim_in, dim_out):
        super(LogisticRegression, self).__init__(dim_in, dim_out)
        self.s = SoftmaxLayer(dim_in, dim_out)
        self.params = self.s.params
        self.s.connect(self.X, self.v)

    def predict(self, x):
        return self.s.predict(x)

    def cost(self, y):
        return self.s.cost(y)
    
    def errors(self, y):
        return self.s.errors(y)

class SimpleMLP(Model):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(SimpleMLP, self).__init__(dim_in, dim_out)
        self.dim_hidden = dim_hidden
        self.layers = [
            Layer(dim_in, dim_hidden),
            SoftmaxLayer(dim_hidden, dim_out)
        ]
        self.layers[0].connect(self.X, self.v)
        self.layers[1].connect(self.layers[0].output, self.layers[0].vec_out)
        self.prediction = self.layers[1].prediction
        self.predicter = theano.function([self.v], self.prediction)
        self.output = self.layers[1].output
        self.params = sum([l.params for l in self.layers],[])

    def predict(self, x):
        return self.predicter(x)

    def cost(self, y):
        return self.layers[1].cost(y)

    def errors(self, y):
        return self.layers[1].errors(y)

class MLP(Model):
    def __init__(self, dim_in, dim_out, layers):
        super(MLP, self).__init__(dim_in, dim_out)
        self.n_layers = len(layers)
        self.layers = layers
        self.layers[0].connect(self.X, self.v)
        for i in range(1,len(layers)):
            layers[i].connect(self.layers[i-1].output, self.layers[i-1].vec_out)
        self.prediction = self.layers[self.n_layers - 1].prediction
        self.predicter = theano.function([self.v], self.prediction)
        self.output = self.layers[self.n_layers - 1].output
        self.params = sum([l.params for l in self.layers],[])
        self.L1 = T.sum([abs(param).sum() for param in self.params])
        self.L2 = T.sqrt(T.sum([(param ** 2).sum() for param in self.params ]))

    def predict(self, x):
        return self.predicter(x)

    def cost(self, y):
        return self.layers[self.n_layers - 1].cost(y)

    def errors(self, y):
        return self.layers[self.n_layers - 1].errors(y)

class NeuralNetwork(object):
    """
    A class for a general neural network model. Supports multiple input layers 
    and multiple output layers. Initialization is slightly more complicated
    to allow for this flexibility, and as a consequence it doesn't extend
    the earlier 'Model' OOP, as this was unintentionally univariate-chauvinistic.
    """
    def __init__(self, dims_in, dims_out, connection_spec, 
                 Xs = None, vs = None):
        """
    
        :type dims_in: list
        :param dims_in: A list of dimensions of input vectors. 
        
        :type dims_out: list
        :param dims_out: A list of dimensions of output vectors.

        :type connection_spec: layers.ConnectionSpec
        :param connection_spec: An object that specifies the layers and connections.
        """
        self.dims_in = dims_in
        self.dims_out = dims_out
        if Xs is None:
            self.Xs = [T.dmatrix() for _ in dims_in]
        else:
            self.Xs = Xs
        if vs is None:
            self.vs = [T.dvector() for _ in dims_in]            
        else:
            self.vs = vs
        print "[NN Init] Copying State from CS to NN."
        self.layers = connection_spec.layers
        self.input_layers = connection_spec.input_layers
        if not(len(self.Xs) == len(self.vs) == len(self.input_layers)):
            raise RuntimeError("The number of inputs do not agree")

        # Assumes everything has been ordered correctly.
        print "[NN Init] connecting up layers"
        for l, X, v in zip(self.input_layers, self.Xs, self.vs):
            l.connect(X,v)
        connection_spec.connecter()
        self.output_layers = connection_spec.output_layers
        self.params = sum([l.params for l in self.layers],[])
        self.L1 = T.sum([abs(param).sum() for param in self.params])
        self.L2 = T.sqrt(T.sum([(param ** 2).sum() for param in self.params ]))
        #self.prediction = theano.scan(lambda x: x.prediction, sequences=self.output_layers)
        print "[NN Init] Compiling Theano Functions."
        self.predicter = theano.function(self.vs, [x.prediction for x in self.output_layers])


    # Assumes vectors given in same order as constructor CS.
    def cost(self, ys):
        if not (len(ys) == len(self.output_layers)):
            raise RuntimeError("Number of outputs does not agree.")
        return T.sum([l.cost(y) for (l,y) in zip(self.output_layers, ys)])

    # Assumes vectors given in same order as constructor CS.
    def errors(self, ys):
        if not (len(ys) == len(self.output_layers)):
            raise RuntimeError("Number of outputs does not agree.")
        return [l.errors(y) for (l,y) in zip(self.output_layers, ys)]

class SkipGram(NeuralNetwork):
    def __init__(self, sentences, vec_size = 50, context_size = 1):
        self.vec_size = vec_size
        self.context_size = context_size
        print "[Skipgram Init] Tokenizing %d sentences" %len(sentences)
        t = text_util.Tokenizer(sentences)
        print "[Skipgram Init] done. Building connection spec."
        n_vocab = t.n_tokens
        layers = [Layer(n_vocab, vec_size)] + [SoftmaxLayer(vec_size, n_vocab) 
                                               for _ in range(2*context_size)]
        connects = {0: range(1,len(layers))}
        cs = ConnectionSpec(layers, connects)
        dims_in = [n_vocab]
        dims_out = [n_vocab for _ in range(context_size*2)]
        self.tokenizer = t
        
        print "[Skipgram Init] done. Doing regular NN init."
        Xs = [T.imatrix() for _ in dims_in]
        vs = [T.ivector() for _ in dims_in]
        super(SkipGram, self).__init__(dims_in, dims_out, cs, Xs, vs)

    def __str__(self):
        return "<SkipGram Model> with vocab size %d, context size %d, vec size %d" %(self.t.n_tokens, self.context_size, self.vec_size)

# TODO: Fit this into the framework of my models. 
class SimpleRNN(Model):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(SimpleRNN, self).__init__(dim_in, dim_out)
        self.dim_hidden = dim_hidden
        self.l = RecurrentLayer(dim_in, dim_hidden, dim_out)
        self.S = T.matrix()
        self.s = T.vector()
        self.h_init = T.vector()

        def step(s_current, h_prev, W_ih, W_hh, W_ho):
            h_t = T.tanh(T.dot(s_current, W_ih) + T.dot(h_prev, W_hh))
            y_t = T.tanh(T.dot(h_t, W_ho))
            return h_t, y_t
    
        [self.H, self.Y], _ = theano.scan(
            step,
            sequences = self.S,
            outputs_info = [self.h_init, None],
            non_sequences = [self.l.W_in, self.l.W_hidden, self.l.W_out]
        )
        
    def error(self, t):
        return ((self.Y - t) ** 2).sum()

    def predict(self, v, h0):
        return self.predicter(v, h0)
    

def test():
    patents = text_util.getpats(1000)
    vec_size, context_size = (20, 2)
    sg = SkipGram(patents, vec_size = vec_size, context_size = context_size)
    t = sg.tokenizer
    Xs, Ys, egs = text_util.skipgram_preprocess(patents, t, context_size)
    return patents, sg, Xs, Ys, egs, t

    
