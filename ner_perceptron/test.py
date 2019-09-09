#!/usr/bin/env python
"""
created at: Sun Sep  8 18:50:43 2019
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Testing the Cython codes.
"""

import viterbi
import array
import numpy as np
from collins_perceptron import decode

word_ixs = array.array('i', range(15))
num_tags = 500
weights = np.random.randint(10, size=(50000, num_tags), dtype=int)

print viterbi.decode(word_ixs, num_tags, weights)
print decode(word_ixs, num_tags, weights)
