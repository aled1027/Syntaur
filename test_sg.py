"""
test_sg.py

Make sure that skipgram architecture is working [SPOILER: IT'S NOT]
"""

import text_util
from models import SkipGram
from optimize import multi_sgd
from matplotlib import pyplot as plt
import theano

# Verbose theano output. Comment below out if that's annoying.
# theano.config.compute_test_value = 'warn'

if __name__ == "__main__":
    patents = text_util.getpats(1000)
    vec_size, context_size = (30, 1)
    skipgram = SkipGram(patents, vec_size=vec_size, context_size=context_size)
    t = skipgram.tokenizer
    print "Preprocessing dataset to produce examples"
    Xs, Ys, egs = text_util.skipgram_preprocess(patents, t, context_size)
    print "Received %d training examples" %Xs[0].shape[0]
    errs = multi_sgd(Xs, Ys, skipgram, verbose = True, n_epochs = 100, learning_rate_init = .2, anneal_lr = True, batch_size = 1000)

    plt.scatter(np.arange(len(erros)), errs)
    plt.show()

    """
    rw = t.rand_word()
    rv = text_util.unit_vector(t.n_tokens,t.token2id[rw])
    predicted_ids = skipgram.predicter(rv)
    predicted_words = [t.id2token[int(i)] for i in predicted_ids]
    print "random word: %s" %rw
    print "Predicted prev, next: %s" %str(predicted_words)
    """



    
