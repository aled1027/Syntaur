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
    multi_sgd(Xs, Ys, skipgram, n_epochs = 200, verbose = True)
    return skipgram
    
