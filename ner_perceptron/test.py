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
wObs = np.random.randint(10, size=(500, num_tags), dtype=int)
wTags = np.random.randint(10, size=(num_tags, num_tags), dtype=int)

cy_out = viterbi.decode(word_ixs, num_tags, wObs, wTags)
py_out = decode(word_ixs, num_tags, wObs, wTags)
for a, b in zip(cy_out, py_out):
    assert a == b, "Cython out != Python out."

times = timeit.Timer(partial(decode, word_ixs, num_tags, wObs, wTags)).repeat(3, 100)
print "vanilla:", min(times)/100.0
times = timeit.Timer(partial(viterbi.decode, word_ixs, num_tags, wObs, wTags)).repeat(3, 100)
print " cython:", min(times)/100.0


if __name__ == '__main__':
    print 'test.py'
