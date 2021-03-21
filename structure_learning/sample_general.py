#!/usr/bin/env python
"""
created at: Mon Jun  8 18:07:26 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

A script to sample data from a general DAG.
Returns adjacency matrix, causality matrix and data samples.

Defaults to the "toy" problem in `Cheng et. al. 1997`.
All vars are binary.
Probability values are for seeing `1`
```
(A)--->(B)--->(C)--->(E)
        |             |
        '---->(D)--->-'
```
"""


import numpy as np
import cPickle as pickle


if __name__ == '__main__':
    nvars = 5
    names = list('ABCDE')

    adjacency_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=np.int32)
    assert adjacency_matrix.shape == (nvars, nvars)
    for i in range(nvars):
        assert np.alltrue(adjacency_matrix[i] == adjacency_matrix[:, i])

    causality_matrix = np.zeros_like(adjacency_matrix, dtype=np.int32)
    causality_matrix[0, 1] = 1
    causality_matrix[1, 2] = 1
    causality_matrix[1, 3] = 1
    causality_matrix[2, 4] = 1
    causality_matrix[3, 4] = 1

    pindep = {
        0: 0.8
    }

    pdepen = {
        1: {0: [0.3, 0.8]},
        2: {1: [0.2, 0.7]},
        3: {1: [0.3, 0.8]},
        4: {(2, 3): np.array([
            [0.2, 0.7],
            [0.3, 0.8]
        ])}
    }

    # Sample ...
    data = []
    nsamples = 1000
    for i in range(nsamples):
        s = [None]*nvars

        for var, p in pindep.items():
            s[var] = np.random.binomial(1, p)

        for var, parents in sorted(pdepen.items()):
            for cond, enums in parents.items():
                if isinstance(cond, int):
                    p = enums[s[cond]]
                    s[var] = np.random.binomial(1, p)

                else:
                    p = enums[s[cond[0]], s[cond[1]]]
                    s[var] = np.random.binomial(1, p)

        data.append(s)

    print "Printing first 10 rows ..."
    data = np.array(data, dtype=np.int32)
    print data[:10]

    print "Dumping data object ..."
    print "Contains:\nnvars\nnames\nadjacency_matrix\ncausality_matrix\n"\
        "pindep\npdepen\nnsamples\ndata"
    with open('data_sample_general.cpkl', 'w') as fp:
        pickle.dump({
            'nvars': nvars,
            'names': names,
            'adjacency_matrix': adjacency_matrix,
            'causality_matrix': causality_matrix,
            'pindep': pindep,
            'pdepen': pdepen,
            'nsamples': nsamples,
            'data': data
        }, fp)

    print "\nDone."
