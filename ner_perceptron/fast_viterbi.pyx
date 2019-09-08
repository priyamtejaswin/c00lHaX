#!/usr/bin/env python
"""
created at: Sun Sep  8 18:13:42 2019
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Viterbi decoding written in Cython. Vanilla:
```python
def decode(words, ft2ix, tag2ix, ix2tag, weights):
    size = len(words)
    num_feats = len(ft2ix)
    num_tags = len(tag2ix)

    delta, psi = [], []

    list_num_tags = range(num_tags)

    # Stage1
    s1_st = time.time()
    for t, word in enumerate(words):  # `t` is time.
        wix = ft2ix.get(word, None)

        if t == 0:
            # Prev tag is fixed. Current tag (i.e. state)
            # needs to be determined, based on the current
            # word only (prev tag will be None).
            if wix is None:
                continue

            _d = [weights[wix, tix] for tix in list_num_tags]
            _p = np.zeros(num_tags, dtype=int).tolist()

        else:
            # Here, we run the |s|^2 loop.
            # Again, we use the `word` feature only if
            # it is present.
            _d, _p = [], []
            for curr in list_num_tags:  # `curr` represents the current tag.
                temp = [delta[t-1][prev] + weights[prev, curr] + \
                        weights[wix, curr] if wix is not None else 0 \
                        for prev in list_num_tags]

                _d.append(np.max(temp))
                _p.append(np.argmax(temp))

        delta.append(_d)
        psi.append(_p)

    # Stage2
    decoding = []
    qt = np.argmax(delta[-1])
    decoding.append(qt)

    for t in range(size)[::-1][:-1]:
        qt = psi[t][qt]
        decoding.append(qt)

    return decoding, [ix2tag[i] for i in decoding]
```
"""

