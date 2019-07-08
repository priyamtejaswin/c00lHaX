#!/usr/bin/env python
"""
Priyam Tejaswin (tejaswin.p@flipkart.com)
CART style regression tree implementation.
Written in VIM. Brevity was the only option.
"""

import ipdb
import sys
import numpy as np
import matplotlib.pyplot as plt

data = np.linspace(0, 5, 50)
target = np.sin(data)
noise = np.random.normal(np.zeros_like(target), 0.1)
target = target + noise

def sse(x):
    hat = np.mean(x)
    return np.sum(np.square(x - hat))

def find_split(r):
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

def split_region(r, splits):
    "The targets are the last col."
    best_fs = find_split(r)
    if best_fs is False:
        print best_fs
    else:
        print best_fs
        splits.append(best_fs)

        f, s = best_fs
        reg1 = r[r[:, f] < s]
        reg2 = r[r[:, f] >= s]

        split_region(reg1, splits)
        split_region(reg2, splits)

    return splits


combined = np.vstack((data, target)).T
all_splits = split_region(combined, [])

# Vizzz
plt.scatter(data, target, c='black')
cmap = plt.cm.get_cmap('hsv', len(all_splits))

for i, (f,s) in enumerate(all_splits):
    plt.plot([s, s], [-2, 2], c=cmap(i), label='split %d'%i)

plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()

