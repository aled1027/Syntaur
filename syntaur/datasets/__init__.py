import os
import logging
import gzip, cPickle

THIS_DIR, _ = os.path.split(__file__)
MNIST_PATH = '/'.join([THIS_DIR, 'mnist.pkl.gz'])
PATN_PATH = '/'.join([THIS_DIR, 'patents100k.txt.gz'])
STOP_PATH = '/'.join([THIS_DIR, 'englishStop.txt'])

f = gzip.open(MNIST_PATH, 'rb')
mnist = cPickle.load(f)
f.close()

f = gzip.open(PATN_PATH, 'rb')
lines = [line.replace('title:. ','').replace('. abstract:. ','') for line in f]
f.close()
patents = lines[:10000]
del lines

f = open(STOP_PATH, 'r')
stoplist = set(f.read().split('\n'))






