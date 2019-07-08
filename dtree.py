#!/usr/bin/env python
"""
Priyam Tejaswin (tejaswin.p@flipkart.com)
Common Decision Tree base class.
"""

import ipdb
import sys
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(23)


data = np.linspace(0, 5, 50)
target = np.sin(data)
noise = np.random.normal(np.zeros_like(target), 0.1)
target = target + noise

plt.scatter(data, target)
plt.show()

def sse(x):
    hat = np.mean(x)
    return np.sum(np.square(x - hat))

def find_split(r):
    ipdb.set_trace()
    print "Tracing ..."

    x = r[:, :-1]
    y = r[:, -1]

    parent_error = sse(y)
    best_error = sys.maxint
    best_split = None

    n_feats = x.shape[1]
    for f in range(n_feats):
        values = x[:, f]
        for s in values:
            reg1 = r[r[:, f] < s]
            reg2 = r[r[:, f] >= s]
            # A moment of silence for Numpy indexing and silence ...

            if reg1.shape[0]<5 or reg2.shape[0]<5:
                continue

            split_error = sse(reg1[:, -1]) + sse(reg2[:, -1])
            if split_error < best_error:
                best_error = split_error
                best_split = (f, s)

    if best_split is None:
        return False
    else:
        return best_split

def split_region(r):
    "The targets are the last col."
    best_fs = find_split(r)
    return best_fs

combined = np.vstack((data, target)).T
print combined.shape
n1_split = find_split(combined)
print n1_split
plt.scatter(data, target)
plt.plot([n1_split[1], n1_split[1]], [0, 1])
plt.show()

