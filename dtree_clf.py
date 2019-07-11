#!/usr/bin/env python
"""
Priyam Tejaswin (tejaswin.p@flipkart.com)

CART style classifier.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(23)

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

            if reg1.shape[0]<15 or reg2.shape[0]<15:
                continue

            split_error = impurity(reg1[:, -1]) + impurity(reg2[:, -1])
            if split_error < best_error:
                best_error = split_error
                best_split = (f, s)

    if best_split is None:
        return False
    else:
        return best_split

def split_region(r, splits):
    "The targets are the last col."
    best_fs = find_split(r)
    if best_fs is False:
        print best_fs
    else:
        print best_fs
        f, s = best_fs
        f_min, f_max = np.min(r[:, 1 - f]), np.max(r[:, 1 - f])
        splits.append( (best_fs, (f_min, f_max)) )

        reg1 = r[r[:, f] < s]
        reg2 = r[r[:, f] >= s]

        split_region(reg1, splits)
        split_region(reg2, splits)

    return splits

combined = np.concatenate((data, target.reshape(-1, 1)), axis=-1)
all_splits = split_region(combined, [])

# Vizzz
for i in range(len(centers)):
    x, y = data[i*25 : (i+1)*25].T
    plt.scatter(x, y)

for ix, tdata in enumerate(all_splits):
    (f, s), (f_min, f_max) = tdata
    if f == 0:
        plt.plot([s, s], [f_min, f_max], label=ix)
    else:
        plt.plot([f_min, f_max], [s, s], label=ix)

plt.legend()
plt.show()
