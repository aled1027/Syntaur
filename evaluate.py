"""
evaluate.py
"""

import theano.tensor as T
import theano

def which_err(data_xy, model):
    data_x, data_y = data_xy
    y = T.ivector('y')
    geterrs = model.errors(y)
    f = theano.function(
        inputs=[],
        outputs=geterrs,
        givens={
            model.X: data_x,
            y: data_y
        }
    )
    return f()
