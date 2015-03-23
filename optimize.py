"""
optimize.py
"""

import theano.tensor as T
import theano
import time
import numpy as np

def simple_sgd(train_xy, model, n_epochs = 100, batch_size = 600, learning_rate = .13, verbose = False):
    y = T.ivector('y')
    index = T.lscalar('index')
    train_x, train_y = train_xy
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size

    cost = model.cost(y)
    gParams = [T.grad(cost=cost, wrt=param) for param in model.params]
    updates = [(param, param - learning_rate * grad) 
               for (param, grad) in zip(model.params, gParams)]

    train_model = theano.function(
        [index],
        outputs=cost,
        updates=updates,
        givens={
            model.X: train_x[index * batch_size : (index+1) * batch_size],
            y: train_y[index * batch_size : (index+1) * batch_size]
        }
    )

    best_error = 100000
    for epoch_i in xrange(n_epochs):
        for batch_i in xrange(n_train_batches):
            e = train_model(batch_i)
            if verbose:
                print "Epoch %d, batch %d, trainining error %f" %(epoch_i, batch_i, e)
        if e < best_error:
            best_error = e
        print "Epoch: %d, best error: %f" %(epoch_i, best_error)

def get_test(test_set, model):
    y = T.ivector('y')
    test_x, test_y = test_set
    return theano.function(
        inputs = [],
        outputs=model.errors(y),
        givens={
            model.X: test_x,
            y: test_y
        }
    )

def get_validate(valid_set, model):
    y = T.ivector('y')
    valid_x, valid_y = valid_set
    return theano.function(
        inputs = [],
        outputs = model.errors(y),
        givens ={
            model.X: valid_x,
            y: valid_y
        }
    )

def sgd(datasets, model, n_epochs = 100, batch_size = 100, learning_rate = .13,
        L1_reg = 0, L2_reg = 0.001, patience = 50, imp_thresh = .995,
        patience_inc = 2, verbose = False):
    
    # unpack dataset
    train_set, valid_set, test_set = datasets
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set

    # determine number of minibatches
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size

    # declare theano symbolic variables
    index = T.lscalar('index')
    y = T.ivector('y')

    cost = model.cost(y) + (L1_reg * model.L1) + (L2_reg * model.L2)

    # defince funcs for checking performance on test set, valid set
    test_model = theano.function(
        inputs = [index],
        outputs = model.errors(y),
        givens = {
            model.X: test_x[index * batch_size : (index+1) * batch_size],
            y: test_y[index * batch_size : (index+1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs = [index],
        outputs = model.errors(y),
        givens = {
            model.X: valid_x[index * batch_size : (index+1) * batch_size],
            y: valid_y[index * batch_size : (index+1) * batch_size]
        }
    )
    
    gradients = [T.grad(cost, wrt = param) for param in model.params]
    updates = [(param, param - learning_rate * gParam) 
               for (param, gParam) in zip(model.params, gradients)]

    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            model.X: train_x[index * batch_size : (index+1) * batch_size],
            y: train_y[index * batch_size : (index+1) * batch_size]
        }
    )

    valid_freq = min(n_train_batches, patience / 2)
    best_valid_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = time.clock()
    done_looping = False

    print("Training model for %d epochs with minibatch size %d, with validation frequency %d, patience %d"%
          (n_epochs, batch_size, valid_freq, patience)
      )
    epoch_i = 0
    while epoch_i < n_epochs and (not done_looping):
        epoch_i += 1
        print "epoch %d" %epoch_i 
        for batch_i in xrange(n_train_batches):
            minibatch_avg_cost = train_model(batch_i)
            if verbose:
                print "epoch %d, batch %d/%d, cost: %f"  %(epoch_i, batch_i, n_train_batches, minibatch_avg_cost)
            j = epoch_i - 1
            if (epoch_i % valid_freq) == 0:
                # validate 
                valid_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_valid_loss = np.mean(valid_losses)
                print(
                    'epoch %d, minibatch %d/%d, validation error: %f' %
                    (
                        epoch_i, 
                        batch_i, 
                        n_train_batches, 
                        this_valid_loss)
                ) 
            
                if this_valid_loss < best_valid_loss:
                    best_valid_loss = this_valid_loss
                    best_iter = j
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    if this_valid_loss < (best_valid_loss * imp_thresh):
                        patience = max(patience, j + patience_inc)

                    print(
                        'epoch %d, minibatch %d/%d, test err of best model: %f.'%
                        (
                            epoch_i,
                            batch_i,
                            n_train_batches,
                            test_score
                        )
                    )

                        
            if patience < j:
                done_looping = True
                break
            
    end_time = time.clock()
    print (
        'Training Complete: Best validation score: %f obtained at iteration %d with  test score: %f. Time elapsed: %.2fm' %
        (
            best_valid_loss, j, test_score, (end_time - start_time) / 60.
        )
    )
    
    
