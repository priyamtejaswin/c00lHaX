#!/usr/bin/env python
"""
Priyam Tejaswin (tejaswin.p@flipkart.com)

CART style classifier.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

centers = [
[5, 3],
[6, 5],
[2, 5],
[3, 2] ]

data = np.vstack([np.random.multivariate_normal(c, [[0.1, 0], [0, 0.1]], 25) for c in centers])
target = np.hstack([[v for i in range(25)] for v in range(len(centers))])

for i in range(len(centers)):
    x, y = data[i*25 : (i+1)*25].T
    plt.scatter(x, y, label=str(i))

plt.show()

def impurity(x):
    """Using Gini."""
    val_counts = Counter(x).items()
    total = len(x) * 1.0
    pmks = [count/total for label,count in val_counts]
    gini = sum([v*(1 - v) for v in pmks])
    return gini

def find_split(r):
    x = r[:, :-1]
    y = r[:, -1]

    parent_error = impurity(y)
    best_error = sys.maxint
    best_split = None

    n_feats = x.shape[1]
    for f in range(n_feats):
        values = x[:, f]
        for s in values:
            reg1 = r[r[:, f] < s]
            reg2 = r[r[:, f] >= s]
            # A moment of silence for those who are doing
            # this without Numpy indexing and slicing ...

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
