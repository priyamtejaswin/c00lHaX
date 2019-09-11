#!/usr/bin/env python
"""
created at: Sun Sep  8 18:50:43 2019
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Testing the Cython codes.
"""

import timeit
import viterbi
import array
import numpy as np
from functools import partial
from collins_perceptron import decode

word_ixs = array.array('i', range(10))
num_tags = 50
weights = np.random.randint(10, size=(500, num_tags), dtype=int)

print viterbi.decode(word_ixs, num_tags, weights)
print decode(word_ixs, num_tags, weights)[::-1]

times = timeit.Timer(partial(decode, word_ixs, num_tags, weights)).repeat(3, 100)
print "vanilla:", min(times)/100.0
times = timeit.Timer(partial(viterbi.decode, word_ixs, num_tags, weights)).repeat(3, 100)
print " cython:", min(times)/100.0


if __name__ == '__main__':
    print 'test.py'
