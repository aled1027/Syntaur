"""
text_util.py

Utilities for taking sentence data and tokenizing it into appropriate input for 
Neural Networks. 
"""

import re
import string
import multiprocessing as mp
import numpy as np
from nltk.corpus import stopwords as nltk_stops
from itertools import islice

#GLOBAL
PUNC = set(string.punctuation)
PATN_PATH = '/Users/jacobmenick/Desktop/Alife/patents100k.txt'

def take(n, iterable, generator = False):
    """ 
    :rtype: list
    :return: A list containing the first n elements of the iterable.
    
    Get a list of the first n items from an iterable.
    Useful for testing code intended take as input a cursor for all docs.

    :type n: int
    :param n: The number of items we'd like from the collection

    :type iterable: iterable
    :param iterable: some iterable data structure; Anything with a __iter__ and __next__ method.

    :type generator: bool
    :param generator: return a boolean?
    """
    out = (x for x in islice(iterable, n))
    if generator:
        return out
    else:
        return list(islice(iterable, n))

def rmPunc(inString):
    """
    :rtype: str
    :return: inString without any punctuation symbols.

    return the string without any punctuation characters.

    :type inString: str
    :param inString: The strong from which we'd like to remove punctuation symbols.
    """
    return ''.join(char for char in inString if char not in PUNC)

def rmNum(inString):
    """
    :rtype: str
    :return: The input string without any numeric characters.

    return the string without any numeric characters.

    :type inString: str
    :param inString: The string from which we'd like to remove numeric characters.
    """
    return ''.join(char for char in inString if not char.isdigit())

def mysplit(inString):
    """
    :rtype: list
    :return: A list of tokens in the input string, split by any num of spaces.

    return a list of tokens, split by any number of spaces. 

    :type inString: str
    :param inString: A string of tokens to split.
    """
    return re.split(u'\s+|\\\\|/', inString.replace('-',' '))

def rm_from_list(lst, bads = None):
    """
    :rtype: list
    :return: A list, without specified elements. 

    return the list, having removed everything in 'bads'

    :type lst: list
    :param lst: The list from which we'd like to remove certain elements

    :type bads: list
    :param bads: The list of elements we'd like to remove from the input list.

    """
    if bads is not None:
        return [x for x in lst if x not in bads]
    else:
        return [x for x in lst]

def parse(inString, noNum = False, noPunc = True):
    if noNum:
        inString = rmNum(inString)
    if noPunc:
        inString = rmPunc(inString)
    return mysplit(inString.lower())

class Tokenizer(object):
    def __init__(self, sentences = None, stopwords = nltk_stops.words('english'), 
                 noNum = True, noPunc = True, verbose = False, 
                 multi_thread = False):
        self.id2token = {}
        self.token2id = {}
        self.docfreqs = {}
        self.n_tokens = 0
        self.stopwords = stopwords
        self.noNum = noNum
        self.noPunc = noPunc


        if verbose: 
            print "Tokenizing %d documents. " %len(sentences)

        # first token is out_of_vocab. 
        # second token is space. 

        self.token2id['OUT_OF_VOCAB'] = 0
        self.id2token[0] = 'OUT_OF_VOCAB'
        self.token2id['END_OF_SENTENCE'] = 1
        self.id2token[1] = 'END_OF_SENTENCE'

        if sentences is not None:
            if not multi_thread:
                i = 2
                for sentence in sentences:
                    words = parse(sentence, noNum = self.noNum, noPunc = self.noPunc)
                    if self.stopwords is not None:
                        words = rm_from_list(words, self.stopwords)
                    for word in words:
                        if word not in self.token2id:
                            if verbose:
                                if (i %1000) == 0:
                                    print "Adding %dth new word, %s" %(i, word)

                            self.token2id[word] = i
                            self.id2token[i] = word
                            self.docfreqs[word] = 1
                            i += 1
                        else: 
                            self.docfreqs[word] += 1
                    self.n_tokens = i
            else:
                raise RuntimeError("Multithreaded tokenizing not yet supported.")
        else:
            self.n_tokens = 2

    def add_more_docs(self, sentences):
        n = self.n_tokens
        for sentence in sentences:
            words = parse(sentence, noNum = self.noNum, noPunc = self.noPunc)
            if self.stopwords is not None:
                words = rm_from_list(words, self.stopwords)
            for word in words:
                if word not in self.token2id:
                    if verbose:
                        if (n %1000) == 0:
                            print "Adding %dth new word, %s" %(n, word)
                    self.token2id[word] = n
                    self.id2token[n] = word
                    self.docfreqs[word] = 1
                    n += 1
                else: 
                    self.docfreqs[word] += 1
        self.n_tokens = n

    def doc2ids(self, inString, update = False):
        bow = []
        words = parse(inString, noNum = self.noNum, noPunc = self.noPunc)
        if self.stopwords is not None:
            words = rm_from_list(words, self.stopwords)
        for word in words:
            if word not in self.token2id:
                if update:
                    self.token2id[word] = self.n_tokens
                    self.id2token[self.n_tokens] = word
                    self.docfreqs[word] = [1]
                    self.n_tokens += 1
                    bow.append(self.token2id[word])
                else:
                    bow.append(self.token2id['OUT_OF_VOCAB'])
            else:
                bow.append(self.token2id[word])
        return bow

    def ids2words(self, bow):
        return [self.id2token[num] for num in bow]

    def transform(self, sentences, multi_thread = False):
        if not multi_thread:
            return [self.doc2ids(sentence) for sentence in sentences]
        else:
            raise RuntimeError("Multithreading not yet supported.")

    def parse_training_eg(self, eg):
        N = len(eg[1])
        bow = eg[1][:N/2] + [eg[0]] + eg[1][(N/2):]
        return map(lambda x: self.id2token[x], bow)

    def rand_word(self, unit_vector = False):
        idx = np.random.randint(low=0, high=self.n_tokens)
        if unit_vector:
            return unit_vector(self.n_tokens, idx)
        else:
            return self.id2token[idx]
    
def get_training_examples(sentences, tokenizer, context=1, backward = True):
    examples = []
    transformed = tokenizer.transform(sentences)    
    for (i, doc) in enumerate(transformed):
        if (i%1000) == 0:
            print "[get_training_examples] on %dth e.g." %i
        for i, word in enumerate(doc):
            c = doc[(i-context):i] + doc[(i+1):(i+1+context)]
            if len(c) == 2 * context:
                examples.append([word, c])
    return examples


                    
def quick_load_patents(patnpath, n = 1000):
    f = open(patnpath, 'r')
    lines = [line.replace('title:. ','').replace('. abstract:. ','') for line in f]
    f.close()
    return take(n, lines)

def getpats(n=1000):
    return quick_load_patents(PATN_PATH, n)

def unit_vector(n, i):
    x = np.zeros(n)
    x[i] = 1
    return x

def skipgram_preprocess(sentences, t, context_size = 1):
    W = t.n_tokens
    print "[Skipgram Preprocess] getting training exmaples"
    egs = get_training_examples(sentences, t, context_size)
    print "[Skipgram Preprocess] Getting Xs from training exs..."
    #Xs = [np.asarray(map(lambda x: unit_vector(W, x[0]), egs))] 
    Xs = [np.asarray(map(lambda x: x[0], egs))] 
    print "[Skipgram Preprocess] Getting Ys from training exs..."
    Ys = [np.asarray(map(lambda x: x[1][i], egs)) for i in range(context_size*2)]
    print "[Skipgram Preprocess] Done."
    return Xs, Ys, egs


def test(n = 1000):
    patents = getpats(n)
    t = Tokenizer(patents)
    egs = get_training_examples(patents, t, context=1)


    


