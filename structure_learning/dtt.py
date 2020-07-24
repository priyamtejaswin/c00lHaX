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
import json
import time
import numpy as np
import networkx as nx
import cPickle as pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from collections import defaultdict
from itertools import combinations


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print '%r  %2.2f s' % \
                (method.__name__, te - ts)
        return result
    return timed


EPSILON = 0.003


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
    uniq_a = set(data[:, a].tolist())
    uniq_b = set(data[:, b].tolist())
    
    for i in uniq_a:
        for j in uniq_b:
            key = (i, j)
            if key in p_joint:
                jval = p_joint[key]/float(nsamples)
                mval_a = p_a[i]/float(nsamples)
                mval_b = p_b[j]/float(nsamples)

                siter = jval * (np.log(jval) - (np.log(mval_a) + np.log(mval_b)))
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

        siter = p_j * (np.log(p_j) + np.log(p_c) - np.log(p_a) - np.log(p_b))
        score += siter

    return score


def get_nbrs_to_sep(G, u, v, parents=None):
    """
    Done TODO: Add support to delete nodes from Nbrs_u, Nbrs_v,
    if they are KNOWN causal children of u, v.
    """
    Nbrs_u, Nbrs_v = set(), set()
    for path in nx.all_simple_paths(G, u, v):
        Nbrs_u.add(path[1])
        Nbrs_v.add(path[-2])

    if parents is not None:
        for ix, val in enumerate(parents[:, u].tolist()):
            if val == 1:
                Nbrs_u.discard(ix)
        for ix, val in enumerate(parents[:, v].tolist()):
            if val == 1:
                Nbrs_v.discard(ix)

    if len(Nbrs_u) > len(Nbrs_v):
        Nbrs_u, Nbrs_v = Nbrs_v, Nbrs_u

    return Nbrs_u, Nbrs_v


@timeit
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


def plot_graph(G, parents=None):
    # Plotting code.
    pos = nx.spring_layout(G, iterations=10)
    nx.draw(G, pos, alpha=0.4, font_size=100, with_labels=False)
    node_labels = {n:v['name'] for n,v in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
    edge_lables = {(e[0], e[1]):round(e[2]['mi_score'], 3) for e in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_lables)
    nx.draw_networkx_edges(G, pos=pos, arrows=False)

    if parents is not None:
        edlist = []
        for i in range(parents.shape[0]):
            for j in range(parents.shape[1]):
                if parents[i, j] == 1:
                    edlist.append((i, j))

        nx.draw_networkx_edges(G, pos=pos, edgelist=edlist, arrows=True, edge_color='b')

    plt.show()


def orient_edges(G, data, nvars):
    """
    Return a matrix for parents, and a collection of undecideds.
    Please, try not to update the Graph...
    """
    parents = np.zeros((nvars, nvars), dtype=np.int32)
    # Column is PARENT OF Row in `parents` matrix.
    undecided = set()

    for a in tqdm(G.nodes()):
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

            if len(N_a) == 0:
                print "Empty set."
                raise RuntimeError("For triplet <s1, a, s2>(%d, %d, %d) set N_a is empty!"%(s1, a, s2))
            if len(N_a) == 1:
                print "Single element in set.", (s1, a, s2)
                # undecided.add((s1, a, s2))
                if cond_mi(data, s1, s2, N_a) > nocond_mi(data, s1, s2)+EPSILON:
                    parents[a, s1] = 1
                    parents[a, s2] = 1
                continue

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

    # Check all triples ...
    valid_triples = set()
    for edge in G.edges:
        u, v = edge
        for n in G.neighbors(u):
            if n != v:
                valid_triples.add((n, u, v))
        for n in G.neighbors(v):
            if n != u:
                valid_triples.add((u, v, n))

    for a,b,c in valid_triples:
        if parents[b, a] == 1 and parents[b, c] == 0 and parents[c, b] == 0 and (a, b, c) not in undecided:
            parents[c, b] = 1
        
    return parents, undecided
        

def main(data, names):
    nvars = data.shape[1]
    assert nvars == len(names)

    G = nx.Graph()
    for i in range(nvars):
        G.add_node(i, name=names[i])
    
    print "Num vars: %d, nC2: %d"%(nvars, nvars*(nvars-1)/2)

    if 'chow_liu_draft_list.cpkl' not in os.listdir(os.getcwd()):
        pairwise_mi_scores = []
        for a, b in tqdm(combinations(range(nvars), 2)):
            mi_score = nocond_mi(data, a, b)
            if mi_score > EPSILON:
                pairwise_mi_scores.append(((a, b), mi_score))

        L = sorted(pairwise_mi_scores, reverse=True, key=lambda x: x[1])
        for row in L:
            print row

        with open('chow_liu_draft_list.cpkl', 'w') as fp:
            pickle.dump(L, fp)
    else:
        with open('chow_liu_draft_list.cpkl') as fp:
            L = pickle.load(fp)
    
    # Start drafting.
    _first = L.pop(0)
    G.add_edge(_first[0][0], _first[0][1], mi_score=_first[1])

    _second = L.pop(0)
    G.add_edge(_second[0][0], _second[0][1], mi_score=_second[1])

    nedges = 2
    ptr = 0
    sizeL = len(L)
    while ptr<sizeL and L and (nedges < nvars-1):
        cedge = L[ptr]
        u, v = cedge[0]
        score = cedge[1]

        if nx.has_path(G, u, v):
            ptr += 1
        else:
            G.add_edge(u, v, mi_score=score)
            nedges += 1
            L.pop(ptr)
            sizeL -= 1

    # Drafting complete.
    print "Drafting complete ..."
    # plot_graph(G)

    # Start thickening.
    ptr = 0
    remain = len(L)
    # _pbar = tqdm(total=remain)
    while ptr < remain:
        cedge = L[ptr]
        u, v = cedge[0]
        score = cedge[1]

        print "Candidate:", ptr, cedge

        N1, N2 = get_nbrs_to_sep(G, u, v)

        if try_to_separate(u, v, data, N1):
            ptr += 1
        else:
            # The first time it fails, re-try with N2.
            if (N1 != N2) and try_to_separate(u, v, data, N2):
                ptr += 1
            else:
                G.add_edge(u, v, mi_score=score)
                ptr += 1

        print

        # _pbar.update(1)
    # _pbar.close()

    # Thickening complete.
    print "Thickening complete ..."
    plot_graph(G)

    # Start orientation.
    parents, undecided = orient_edges(G, data, nvars)

    # Orientation complete.
    diG = nx.DiGraph()
    for _n, _d in G.nodes(data=True):
        diG.add_node(_n, **_d)
    for etup in G.edges.data():
        _u, _v, _d = etup
        diG.add_edge(_u, _v, **_d)

    plot_graph(diG, parents)

    print parents

    # Start thinning.
    all_edges = list(G.edges.data())
    for etup in tqdm(all_edges):
        u, v, d = etup
        if len(list(nx.all_simple_paths(G, u, v))) > 1:
            # Temporarily delete this edge ...
            G.remove_edge(u, v)

            # Try to separate ...
            N1, N2 = get_nbrs_to_sep(G, u, v, parents=parents)
            if len(N1) == 0 and len(N2) == 0:
                G.add_edge(u, v, **d)
                continue

            if len(N1)>0 and try_to_separate(u, v, data, N1):
                pass
            else:
                # The first time it fails, re-try with N2.
                if len(N2)>0 and try_to_separate(u, v, data, N2):
                    pass
                else:
                    G.add_edge(u, v, **d)

    # Thinning complete.
    plot_graph(G)

    # Export to viz in D3.js
    nodes, links = [], []
    for _n, _d in G.nodes(data=True):
        nodes.append({
            'name': _d['name'],
            'label': str(_n),
            'id': _n
        })
    for _u, _v, _d in G.edges(data=True):
        if parents[_v, _u] == 1:
            directed = True
        elif parents[_u, _v] == 1:
            _v, _u = _u, _v
            directed = True
        else:
            directed = False
        links.append({
            'source': _u,
            'target': _v,
            'value': _d['mi_score'],
            'directed': directed
        })
    with open('output.json', 'w') as fp:
        fp.write(json.dumps({'nodes':nodes, 'links':links}, indent=4, separators=(',', ': ')))


if __name__ == '__main__':
    # # path = sys.argv[1]
    # path = 'data_sample_general.cpkl'
    # assert os.path.exists(path)

    # with open(path) as fp:
    #     data = pickle.load(fp)
    #     print data.keys()

    # main(data['data'], data['names'])

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    lbe = LabelEncoder()
    raw = pd.read_csv(sys.argv[1], index_col=None, dtype=str)
    main(raw.to_numpy()[:10000], raw.columns)
    
    # encoded = {c:lbe.fit_transform(raw[c]) for c in raw.columns}
    # alarm = pd.DataFrame.from_dict(encoded)

    # print alarm.shape
    # main(alarm.to_numpy(), alarm.columns)
