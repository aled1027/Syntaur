# Syntaur

A bare bones neural networks library for python built with theano.
Aims to provide maximal flexibility. 

```
import syntaur
from optimize import multi_sgd
from models import SkipGram, MLP
from datasets import patents, mnist
from util import visualize_image, shared_dataset

# Fit MLP model on mnist handwritten digit dataset
mnist_test, _, _ = mnist
mnist_shared = map(shared_dataset, mnist)

dim_in, dim_hidden1, dim_hidden2, dim_out = (28*28, 200, 100, 10)
mlp_layers = [
  Layer(dim_in, dim_hidden1),
  Layer(dim_hidden1, dim_hidden2),
  SoftmaxLayer(dim_hidden3, dim_out)
]

mlp = MLP(dim_in, dim_out, layers)

# Train with stochastic gradient descent
uni_sgd(mnist_shared, mlp, n_epochs = 5, verbose = False, patience = 50)

# Check to see how our model does on unseen data. 
test_x, test_y = mnist_test
random_index = np.random.randint(low=0, high=test_x.shape[0])
random_image = test_x[random_index]
correct_label = test_y[random_index]
predicted = int(mlp.predict(random_image))
print "predicted == correct_label"

# check out the image. 
visualize_image(random_image)


# Fit SkipGram model on patent dataset.

```
