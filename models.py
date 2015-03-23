"""
models.py
"""

from layers import Layer, SoftmaxLayer
import theano
import theano.tensor as T

class Model(object):
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in 
        self.dim_out = dim_out
        
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
        self.X = T.dmatrix('X')
        self.v = T.dvector('v')
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
        self.X = T.dmatrix('X')
        self.v = T.dvector('v')
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
    def __init__(self, dim_in, dim_out, layers, regularized = False):
        super(MLP, self).__init__(dim_in, dim_out)
        self.X = T.dmatrix('X')
        self.v = T.dvector('v')
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




        

        

    
