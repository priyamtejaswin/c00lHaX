#!/usr/bin/env python
"""
created at: Tue Jun  9 14:35:04 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Implements Cheng et. al. solution from 1997:
"Learning Belief Networks from Data: An Information Theory Approach"
DTT => Drafting Thickening Thinning
"""


import os
import sys
import numpy as np
import networkx as nx
import cPickle as pickle
import matplotlib.pyplot as plt
from collections import deque
from collections import defaultdict
from itertools import combinations


def nocond_mi(data, a, b):
    nsamples = data.shape[0]
    p_joint = defaultdict(int)
    p_a = defaultdict(int)
    p_b = defaultdict(int)

    for row in data:
        key_a = row[a]
        p_a[key_a] += 1

        key_b = row[b]
        p_b[key_b] += 1

        key_joint = tuple(row[[a, b]])
        p_joint[key_joint] += 1

    score = 0.0
    for i in [0, 1]:
        for j in [0, 1]:
            key = (i, j)
            if key in p_joint:
                jval = p_joint[key]/float(nsamples)
                mval_a = p_a[i]/float(nsamples)
                mval_b = p_b[j]/float(nsamples)

                siter = jval * (np.log2(jval) - (np.log2(mval_a) + np.log2(mval_b)))
                score += siter

    return score


def cond_mi(data, a, b, condvars):
    return


def main(data, names):
    nvars = data.shape[1]
    assert nvars == len(names)

    G = nx.Graph()
    for i in range(nvars):
        G.add_node(i, name=names[i])

    pairwise_mi_scores = []
    for a, b in combinations(range(nvars), 2):
        mi_score = nocond_mi(data, a, b)
        print a, b, mi_score
        pairwise_mi_scores.append(((a, b), mi_score))

    L = deque(sorted(pairwise_mi_scores, reverse=True, key=lambda x: x[1]))
    
    # Start drafting.
    _first = L.popleft()
    G.add_edge(_first[0][0], _first[0][1], mi_score=_first[1])

    _second = L.popleft()
    G.add_edge(_second[0][0], _second[0][1], mi_score=_second[1])

    nedges = 2
    # while L and nedges < nvars-1:
    #     cedge = L[0]
    #     v, a = cedge[0]

    # Plotting code.
    pos = nx.spring_layout(G, iterations=10)
    nx.draw(G, pos, alpha=0.4, font_size=16, with_labels=True)
    edge_lables = {(e[0], e[1]):round(e[2]['mi_score'], 3) for e in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_lables)
    nx.draw_networkx_edges(G, pos=pos)
    plt.show()


if __name__ == '__main__':
    # path = sys.argv[1]
    path = 'data_sample_general.cpkl'
    assert os.path.exists(path)

    with open(path) as fp:
        data = pickle.load(fp)
        print data.keys()

    main(data['data'][:10], data['names'])