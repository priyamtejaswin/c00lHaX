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


EPSILON = 1e-2


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
    if len(condvars) == 0:
        raise NotImplementedError("Should not be computing ConditionalMI with empty Cvars.")
    nsamples = data.shape[0]

    all_vars = [a, b]
    if isinstance(condvars, set):
        all_vars += list(condvars)
    elif isinstance(condvars, list):
        all_vars += condvars
    else:
        raise TypeError('`condvars` is not a list or a set.')

    subset = data[:, all_vars]
    joint = defaultdict(int)

    for row in subset:
        key_joint = tuple(row.tolist())
        joint[key_joint] += 1
    
    marg_a = defaultdict(int)
    marg_b = defaultdict(int)
    marg_c = defaultdict(int)
    for k, v in joint.items():
        k = list(k)

        key_a = tuple([k[0]] + k[2:])
        marg_a[key_a] += v

        key_b = tuple([k[1]] + k[2:])
        marg_b[key_b] += v

        key_c = tuple(k[2:])
        marg_c[key_c] += v

    score = 0.0
    for k, v in joint.items():
        k = list(k)

        p_j = v/float(nsamples)

        key_a = tuple([k[0]] + k[2:])
        p_a = marg_a[key_a]/float(nsamples)

        key_b = tuple([k[1]] + k[2:])
        p_b = marg_b[key_b]/float(nsamples)

        key_c = tuple(k[2:])
        p_c = marg_c[key_c]/float(nsamples)

        siter = p_j * (np.log2(p_j) + np.log2(p_c) - np.log2(p_a) - np.log2(p_b))
        score += siter

    return score


def get_nbrs_to_sep(G, u, v):
    """
    TODO: Add support to delete nodes from Nbrs_u, Nbrs_v,
    if they are KNOWN causal children of u, v.
    """
    Nbrs_u, Nbrs_v = set(), set()
    for path in nx.all_simple_paths(G, u, v):
        Nbrs_u.add(path[1])
        Nbrs_v.add(path[-2])

    if len(Nbrs_u) > len(Nbrs_v):
        Nbrs_u, Nbrs_v = Nbrs_v, Nbrs_u

    return Nbrs_u, Nbrs_v


def try_to_separate(u, v, data, Cvars):
    # Conduct CI test using Nbrs_u
    if len(Cvars) == 0:
        raise NotImplementedError("No conditional variables are present ...")
    else:
        root_v = cond_mi(data, u, v, Cvars)
        if root_v < EPSILON:
            return True
        else:
            if len(Cvars) > 1:
                while len(Cvars) > 0:
                    possible_vs = []
                    for i in Cvars:
                        possible_vs.append((cond_mi(data, u, v, Cvars-{i}), i))

                    least = sorted(possible_vs)[0]
                    minimum_v = least[0]
                    minimum_Cvars = Cvars - {least[1]}

                    if minimum_v < EPSILON:
                        return True
                    elif minimum_v > root_v:
                        return False
                    else:
                        root_v = minimum_v
                        Cvars = minimum_Cvars
                    
                    if len(Cvars) == 1:  # If only 1 var, then this is the `minimum_v`
                        return False
                    
            else:
                return False


def plot_graph(G):
    # Plotting code.
    pos = nx.spring_layout(G, iterations=10)
    nx.draw(G, pos, alpha=0.4, font_size=16, with_labels=True)
    edge_lables = {(e[0], e[1]):round(e[2]['mi_score'], 3) for e in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_lables)
    nx.draw_networkx_edges(G, pos=pos)
    plt.show()


def orient_edges(G, data, nvars):
    """
    Return a matrix for parents, and a collection of undecideds.
    Please, try not to update the Graph...
    """
    parents = np.zeros((nvars, nvars), dtype=np.int32)
    # Column is PARENT OF Row in `parents` matrix.
    undecided = set()

    for a in G.nodes():
        Sa = list(G.neighbors(a))
        for s1, s2 in combinations(Sa, 2):
            N_a = set()
            for path in nx.all_simple_paths(G, s1, s2):
                N_a.add(path[1])
                N_a.add(path[-2])
            # WARNING! all_simple_paths will include edge s1-s2 as well!!!
            # Hence, you have to discard.
            N_a.discard(s1)
            N_a.discard(s2)

            mi_with_a = cond_mi(data, s1, s2, N_a)
            mi_wout_a = cond_mi(data, s1, s2, N_a-{a})

            if mi_with_a > mi_wout_a+EPSILON:
                parents[a, s1] = 1
                parents[a, s2] = 1

            # TODO: This can't be an exact match ...
            if mi_wout_a+EPSILON >= mi_with_a > mi_wout_a:
                undecided.add((s1, a, s2))

        ix_prnts = [i for i in range(nvars) if parents[a, i]==1]
        for x in Sa:
            if parents[a, x] == 0:  # NOT a `parent` of `a`.
                for p in ix_prnts:
                    if (x, a, p) not in undecided:
                        parents[x, a] = 1

    return parents, undecided
        

def main(data, names):
    nvars = data.shape[1]
    assert nvars == len(names)

    G = nx.Graph()
    for i in range(nvars):
        G.add_node(i, name=names[i])

    pairwise_mi_scores = []
    for a, b in combinations(range(nvars), 2):
        mi_score = nocond_mi(data, a, b)
        pairwise_mi_scores.append(((a, b), mi_score))

    L = sorted(pairwise_mi_scores, reverse=True, key=lambda x: x[1])
    for row in L:
        print row
    
    # Start drafting.
    _first = L.pop(0)
    G.add_edge(_first[0][0], _first[0][1], mi_score=_first[1])

    _second = L.pop(0)
    G.add_edge(_second[0][0], _second[0][1], mi_score=_second[1])

    nedges = 2
    ptr = 0
    while L and (nedges < nvars-1):
        cedge = L[ptr]
        u, v = cedge[0]
        score = cedge[1]

        if nx.has_path(G, u, v):
            ptr += 1
        else:
            G.add_edge(u, v, mi_score=score)
            nedges += 1
            L.pop(ptr)

    # Drafting complete.
    print "Drafting complete ..."
    plot_graph(G)

    # Start thickening.
    ptr = 0
    remain = len(L)
    while ptr < remain:
        cedge = L[ptr]
        u, v = cedge[0]
        score = cedge[1]

        N1, N2 = get_nbrs_to_sep(G, u, v)

        if try_to_separate(u, v, data, N1):
            ptr += 1
        else:
            # The first time it fails, re-try with N2.
            if try_to_separate(u, v, data, N2):
                ptr += 1
            else:
                G.add_edge(u, v, mi_score=score)
                ptr += 1

    # Thickening complete.
    plot_graph(G)

    # Start orientation.
    orient_edges(G, data, nvars)





if __name__ == '__main__':
    # path = sys.argv[1]
    path = 'data_sample_general.cpkl'
    assert os.path.exists(path)

    with open(path) as fp:
        data = pickle.load(fp)
        print data.keys()

    main(data['data'], data['names'])