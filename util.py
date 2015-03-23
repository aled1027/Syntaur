"""
util.py
"""

import gzip, cPickle
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

# GLOBAL
MNPATH = '/Users/jacobmenick/Desktop/sandbox/learn_theano/data/mnist.pkl.gz'

def load_mnist(dataset_path):
    f = gzip.open(dataset_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(
        np.asarray(data_x, dtype = theano.config.floatX),
        borrow=True
    )

    shared_y = theano.shared(
        np.asarray(data_y, dtype = theano.config.floatX),
        borrow=True
    )
    return shared_x, T.cast(shared_y, 'int32')

def quick_load_mnist():
    return map(shared_dataset, load_mnist(MNPATH))

def visualize_image(pixel_vector):
    n = np.sqrt(pixel_vector.shape[0])
    if not n.is_integer():
        raise RuntimeError("Image is not square. ")
    pixel_array = np.array(np.split(pixel_vector, n))
    plt.imshow(pixel_array)
    plt.show()
