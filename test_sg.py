"""
test_sg.py

Make sure that skipgram architecture is working
"""

import text_util
from models import SkipGram
from optimize import multi_sgd

def main():
    patents = text_util.getpats(1000)
    vec_size, context_size = (30, 2)
    skipgram = SkipGram(patents, vec_size=vec_size, context_size=context_size)
    t = skipgram.tokenizer
    Xs, Ys, egs = text_util.skipgram_preprocess(patents, t, context_size)
    multi_sgd(Xs, Ys, skipgram, verbose = True, n_epochs = 300)
    rw = t.rand_word()
    rv = text_util.unit_vector(t.n_tokens,t.token2id[rw])
    predicted_ids = skipgram.predicter(rv)
    predicted_words = [t.id2token[int(i)] for i in predicted_ids]
    print "random word: %s" %rw
    print "Predicted prev, next: %s" %str(predicted_words)

main()

    
