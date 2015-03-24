"""
test.py
"""

import util
import random
from models import LogisticRegression, SimpleMLP, MLP
from layers import Layer, SoftmaxLayer
from optimize import simple_sgd, get_test, uni_sgd
from evaluate import which_err
# import multiprocessing as mp
import theano

# GLOBAL
N_SAMPLE = 10
N_EPOCHS = 100
theano.config.exception_verbosity = 'high'

# load data
datasets  = util.quick_load_mnist()
mnist_train, mnist_valid, mnist_test = datasets

# set parameters
dim_in, dim_hidden1, dim_hidden2, dim_hidden3, dim_out = (28*28, 200, 100, 150, 10)

# build layers
layers = [
    Layer(dim_in, dim_hidden1),
    Layer(dim_hidden1, dim_hidden2),
    Layer(dim_hidden2, dim_hidden3),
    SoftmaxLayer(dim_hidden3, dim_out)
]

# Create model instances
mlp = MLP(dim_in, dim_out, layers)

# train models
uni_sgd(datasets, mlp, n_epochs = N_EPOCHS, verbose = False, patience = 50)

test_x, test_y = mnist_test
data = test_x.get_value()
ex = test_x[50]

def test(N_SAMPLE = 10):
    random_elems = random.sample(test_x.get_value(), N_SAMPLE)
    for vec in random_elems:
        mlp_pred = int(mlp.predict(vec))
        print "MLP prediction: %d" %mlp_pred
        util.visualize_image(vec)

def summarize():
    test_model = get_test(mnist_test, mlp)
    mlp_err = test_model()
    print "test err: %f" %mlp_err
    w = which_err(mnist_test, mlp)
    return [i for (i,v) in enumerate(list(w)) if v > 0]

errs = summarize()

"""
for e in errs:
    v = data[e]
    print "prediction: %d" %(int(mlp.predict(v)))
    util.visualize_image(v)
"""
