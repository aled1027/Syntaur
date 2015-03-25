import syntaur
import numpy as np
import matplotlib.pyplot as plt
from syntaur.datasets import patents, mnist, stoplist
from syntaur.util import visualize_image, shared_dataset
from syntaur.text_util import skipgram_preprocess
from syntaur.layers import Layer, SoftmaxLayer
from syntaur.models import SkipGram, MLP
from syntaur.optimize import uni_sgd, multi_sgd


####################################################
# Recognizing MNIST handwritten digits with a MLP
####################################################

# load dataset
mnist_test, _, _ = mnist
mnist_shared = map(shared_dataset, mnist)

# Set parameters
dim_in, dim_hidden1, dim_hidden2, dim_out = (28*28, 200, 100, 10)

# Feedforward net, so layers go in sequence
mlp_layers = [
  Layer(dim_in, dim_hidden1),
  Layer(dim_hidden1, dim_hidden2),
  SoftmaxLayer(dim_hidden2, dim_out)
]

# Get a MLP object.
mlp = MLP(dim_in, dim_out, mlp_layers)

# Train with stochastic (univariate) gradient descent
uni_sgd(mnist_shared, mlp, n_epochs = 2, verbose = True, patience = 50)

# Check to see how our model does on unseen data. 
test_x, test_y = mnist_test

# Choose random index from dataset
random_index = np.random.randint(low=0, high=test_x.shape[0])

# Get random image, label, and model's prediction.
random_image = test_x[random_index]
correct_label = test_y[random_index]
predicted = int(mlp.predict(random_image))
print "predicted: %d, correct label: %d" %(predicted, correct_label)

# check out the image. 
visualize_image(random_image)

# Fit SkipGram model on patent dataset.
sample_patents = patents[:200]

# set skipgram parameters
vec_size, context_size = (20, 3)

# build SkipGram model
skipgram = SkipGram(sample_patents, vec_size = vec_size, context_size = context_size)
t = skipgram.tokenizer

# Preprocess data to get training examples (e.g [w_3, [w_2, w_4]])
Xs, Ys, egs = skipgram_preprocess(sample_patents, t, context_size)

# Train the skipgram net with multivariate stochastic gradient descent
errs = multi_sgd(Xs, Ys, skipgram, verbose = True, n_epochs = 1, learning_rate_init = .2, anneal_lr = True, batch_size = 600)

plt.scatter(np.arange(len(errs)), errs)
plt.show()
