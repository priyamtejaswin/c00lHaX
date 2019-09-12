#!/usr/bin/env python
"""
created at: Sun Aug 25 14:41:52 2019
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

[Collins, "Averaged Perceptron", EMNLP 2002](https://www.aclweb.org/anthology/W02-1001)

Extended the general perceptron algorithm to `structured` predition:
1. Assign weights
2. Run Viterbi Inference
3. Compare predicted tags against truth.
4. Update weights.

### 25th Aug
[14:45] - Paper seems *simple enough*.

[18:25] - Slept. Was sick. Resuming now.
- YT just recommended Deftones' Saturday Night Wrist. Just so good ...
- The paper uses two datasets: Penn TB for POS tagging and a different 
one for base-noun phrase recognition.
- Sticking with the POS tagging task -- unable to find the other dataset.
- Features from [Ratnaparkhi, 1996](https://www.aclweb.org/anthology/W96-0213)
- Read the feature list. Again, seems *simple enough* One issue that I can see 
is that only one or two `tag` related features will be associated with every 
word. Initially, when the model is bad, it will likely be choosing wrong `tag` 
features. My guess is that the feature vector will contain entries against ALL 
the possible tags and only the valid ones will be `1` in the training data. 
The model will start from the incorrect tags and will slowly learn to get to 
the correct ones.
- How does one find the PennTreeBank????

[19:54] - Turns out the PTB costs $1500. Found the Brown corpus instead.
- POS tags used here are very similar to the PTB tags.
- There will be some pre-processing required. I've seen papers refer to some 
specific scripts for this. Will check those after eating. Taking a break now.

[23:37] - Resuming now.
[00:18] - Beat. Looked at the tagset. Will resume tomorrow.

### 1st Sep
[13:50] - Resuming now. Did not get any time this week.
- I was on pre-processing the Brown Corpus data. Could not find a standard 
script online.

[19:29] - Had dozed off in b/w. This needs to stop ...
- Done with data loading and cleaning. Onto feature extraction now.
- Like I mentioned before, I'll have to create placeholders for all possible 
tags.

A feature $f_j(h, t)$ takes in the history and tags to generate a binary value.
For a word as position $i$, the history $h_i = {w_i, w_{i+1}, w_{i+2}, w_{i-1},
 w_{i-2}, t_{i-1}, t_{i-2}}$. We assume the current tag $t_i$ is dependent on 
 $f_j$.

[20:12] - Assuming none are rare and got the features. Now to convert to vector.

### 2nd Sep
[00:09] - Got distracted. Resuming now.
- Trying to wrap my head around the feature representation.
- A feature here is just a <condition,tag> pair. Global feature is sum of 
counts of said local features. Every global feature has a weight (alpha) with 
it. The training procedure updates these alphas.

[01:38] - Trying to generate the feature map/matrix now.
- Running into some *complex* tokens like "5-1/2/cd". The tag here is "cd",i.e. 
cardinal, but splitting on '/' causes a problem. Skipping seqs which contain 
this stuff for now. Will track how many seqs/tokens I skip.
- TryExcept only skips 160 of 45k sequences.

[02:25] - Feature map is ready.

[03:06] - Weights are init now. Will write training loop next.
- Unrelated, but Alien Covenant looks promising so far ... halfway through.
Logging off. Will resume tomorrow.

[13:02] - Resuming now. But before that, here's my short rant about Covenant.
What a disappointment. There was so much potential to explore the story arc 
of the Engineers -- why did they create life and then decided to destroy it? 
why did they stop midway (Prometheus)? It's clear that David wants to create, 
but why destroy the engineers as well? Even the action/horror was not that 
immersive -- it was not really close quarters so there was never a threat to 
anyone on board the Covenant. Hopefully someone will get the bright idea of 
converting this into a tv-show and plug in the gaps. I know a lot of people 
on the *interwebs* who'd pay good money to watch that, including yours truly. 

And now, back to our regular programming. *Pun intended*.
- Writing the main training loop now.

[13:22] - Idiot. Figured out why it has to be a $H \times T$ matrix.

### 3rd Sep
[01:48] - Decoding is tough.
I've been trying to implement Viterbi for Collins' perceptron model. There are 
two issues here. The first is that he says we can use the same `deocding` as 
the Max-Ent models, but that paper uses beam-search, which is a greedy decoder. 

The second problem -- and this is slightly related -- is how we implement 
Viterbi. Beyond a greedy-search (where we fix the current best tag and move on) 
we'll have to treat the previous tag as unknown.

Our $S x T$ matrix will now need to consider all possible <PREV_TAG> features 
as opposed to just considering the ones observed during training.

Will tackle this after I wake up.

Also, Sharp Objects is really addictive.

[02:35] - Logging off.

### 6th Sep
[20:18] - After reading some more, I think I misunderstood the features.
- My feature set is huge because I take a CROSS of the <tag2,tag1,tag> features 
and also the <word,tag> features. Since both can be used to predict the tag, 
I feel that it can be relaxed, i.e. split into two classes. This is also a bit 
similar to the trigram features that was described in section 2.1 of Collins' 
paper.
- To further simplify the algorithm (and my understanding of it), I'll limit
the tag history to just the previous tag. This will also keep the Viterbi
implementation *vanilla* (i.e. 1st order markov).

### 7th Sep - Porier VS Khabib is TONIGHT!
[14:03] - Resmuing the feature generator changes.

[16:44] - Wasted a lot of time in between ... but I wrote the decoder.
- Decoding works. Takes some time (because all possible tags are considered)
but I'm sure there's a way to remove that for loop, or at least move to faster
comprehensions.
- I've accounted for edge cases and stuff.
- Will test with full training routine after having food.

[19:50] - Resuming now. Match starts at 11:30PM.

[21:00] - Wrote the single train step.
- I make it a point NOT to update the param values in-place inside the 
`train_step` function, but to send the changes and update in the training loop.

[21:14] - Main training loop is running.
- But it's slow af. Will try to speed it up. Current rate is ~4s/it
- Going for a run now.

### 8th Sep - Porier lost. GG. Working on speeding up the decoding.
- Maybe i'll try out a Python profiler now?

[17:09] - Tried replacing loops with comprehensions. Some speedups.
But not enough.

[19:03] - Readup on Cython documentation and code examples.
- I'll have to make changes to the decoding function so that it's easier to
work with Ctypes.

[22:39] - DONE.
- Wrote the full decoder in Cython. Stage1 (the biggest culprit) is fully
ported. Stage2 still pending.
- Functionally tested against random data w.r.t. the vanilla decoder.
- Feels like it's 10x faster.  Will integrate with full code tomorrow. Eating
dinner and logging off.

### 9th Sep
[23:33] - Cleaned up the directory. Got the Makefile to work. Updated .gitignore
- Working on integration now.
- *ohyeah* it's working. And it's fast.

### 10th Sep
[23:17] - TIL tqdm let's you print without breaking the progress bar.
- Writing the loop for testing now.

### 12th Sep - I've been debugging this since yesterday. I'm stooopid.
- Features are getting mixed up in the `ix2ft` array and the weights matrix.
- Isolating the word/observation features from the tag/transition features
should solve the problem. Current results are random. It's not learning anything
meaningful. Logging off.

### 13th Sep - [01:29] - Updated the entire code to transition and observation
features. Still no progress. I'm starting to doubt my Viterbi implementation. So
first, I'll ditch Viterbi and implement a greedy approach -- should work well.
I'll also unit-test the Viterbi module with sample data to see if it can decode
the obvious sequences. 
- Logging off.
"""


import os
import sys
import plac
import ipdb
import time
import random
random.seed(23)
import array
import viterbi
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.4f sec' % (method.__name__, te-ts)
        return result

    return timed


def get_clean_line(line):
    line = line.lower().strip()
    if len(line) < 1:
        return None
    else:
        words, tags = [], []

        for token in line.split():
            try:
                word, tag = token.strip().split('/')
            except:
                return False  # Is this allowed?

            if word == tag:  # Ignore.
                pass
            elif tag == '.':  # Sentence closer.
                pass
            else:
                words.append(word)
                tags.append(tag.upper())

        if len(words) < 1:
            return None
        else:
            return words, tags


def get_features(words, tags, w_is_rare=None):
    if w_is_rare is None:
        w_is_rare = [False for x in words]

    size = len(words)
    _pad_false = lambda s: [None] + s

    words = _pad_false(words)
    tags = _pad_false(tags)
    rare = _pad_false(w_is_rare)

    features = []

    for ix in range(1, 1+size):
        if rare[ix] is True:
            pass  # We'll figure this out later.
        else:
            # $h_i = {w_i, w_{i+1}, w_{i+2}, w_{i-1}, w_{i-2}, t_{i-1}, t_{i-2}}
            features.append(((tags[ix-1], words[ix]), tags[ix]))

    return features


def load_files(files_list):
    data = []
    c = 0
    skipped = 0

    for f in tqdm(files_list):
        with open(f, 'r') as fp:
            for line in fp.readlines():
                tokens = get_clean_line(line)
                if tokens is False:
                    skipped += 1
                elif tokens is None:
                    pass
                else:
                    c += 1
                    data.append(tokens)

    print "Found %d sequences."%c
    print "Skipped %d sequences."%skipped
    return data


def get_words_tags_weights(list_of_seq_feats):
    word2ix, ix2word = {}, []
    tag2ix, ix2tag = {}, []
    w, t = 0, 0

    for seq_feats in tqdm(list_of_seq_feats):
        for (prevt, word), tag in seq_feats:
            if prevt not in tag2ix:
                tag2ix[prevt] = t
                ix2tag.append(prevt)
                t += 1

            if tag not in tag2ix:
                tag2ix[tag] = t
                ix2tag.append(tag)
                t += 1

            if word not in word2ix:
                word2ix[word] = w
                ix2word.append(word)
                w += 1

    print "Found %d words."%w
    print "Found %d tags."%t

    return word2ix, ix2word, tag2ix, ix2tag,\
           np.zeros((w, t), dtype=int), np.zeros((t, t), dtype=int)


def get_feature_map_and_weights(list_of_seq_feats):
    ft2ix, ix2ft = {}, []
    tag2ix, ix2tag = {}, []
    f, t = 0, 0

    for seq in tqdm(list_of_seq_feats):
        for feats, tag in seq:
            for val in feats:
                if val not in ft2ix:
                    ft2ix[val] = f
                    ix2ft.append(val)
                    f += 1

            if tag not in tag2ix:
                tag2ix[tag] = t
                ix2tag.append(tag)
                t += 1

            if tag not in ft2ix:
                ft2ix[tag] = f
                ix2ft.append(tag)
                f += 1

    print "Found %d features."%f
    print "Found %d tags."%t

    weights = np.zeros((f, t), dtype=int)

    return ft2ix, ix2ft, tag2ix, ix2tag, weights


# @timeit
def decode(words_ixs, num_tags, wObs, wTags):
    # `word_ixs` is the indices of the word, already mapped.
    # `-1` means OOV.
    size = len(words_ixs)

    delta = np.zeros((size, num_tags), dtype=float)
    psi = np.zeros((size, num_tags), dtype=int)

    list_num_tags = range(num_tags)

    # Stage1
    for t, wix in enumerate(words_ixs):  # `t` is time.
        if t == 0:
            # Prev tag is fixed. Current tag (i.e. state)
            # needs to be determined, based on the current
            # word only (prev tag will be None).
            if wix is -1:
                continue

            for tix in list_num_tags:
                delta[t, tix] = wObs[wix, tix]

            #_d = [weights[wix, tix] for tix in list_num_tags]
            #_p = np.zeros(num_tags, dtype=int).tolist()

        else:
            # Here, we run the |s|^2 loop.
            # Again, we use the `word` feature only if
            # it is present.

            #_d, _p = [], []
            for curr in list_num_tags:  # `curr` represents the current tag.
                temp = [delta[t-1][prev] + wTags[prev, curr] + \
                        wObs[wix, curr] if wix != -1 else 0 \
                        for prev in list_num_tags]

                delta[t, curr] = np.max(temp)
                psi[t, curr] = np.argmax(temp)

                #_d.append(np.max(temp))
                #_p.append(np.argmax(temp))

        #delta.append(_d)
        #psi.append(_p)

    # Stage2
    decoding = []
    qt = np.argmax(delta[-1])
    decoding.append(qt)

    for t in range(size)[::-1][:-1]:
        qt = psi[t][qt]
        decoding.append(qt)

    return decoding


# @timeit
def forward(wdseq, tgseq, word2ix, ix2tag, wObs, wTags):
    # Given word seq, get the predicted tag seq.
    # `deixs` -- tag indices.
    # `deseq` -- tag strings.
    # `wdixs` -- word indices.
    wdixs = [word2ix.get(w, -1) for w in wdseq]
    wdixs = array.array('i', wdixs)
    deixs = decode(wdixs, len(ix2tag), wObs, wTags)
    deseq = [ix2tag[i] for i in deixs]

    # Extract features from truth and pred pairs.
    gold_feats = get_features(wdseq, tgseq)
    pred_feats = get_features(wdseq, deseq)

    return gold_feats, pred_feats


def train_step(gold_feats, pred_feats, word2ix, tag2ix):
    obs_pos = defaultdict(int)
    tag_pos = defaultdict(int)

    for (prevt, word), tag in gold_feats:
        c = tag2ix[tag]
        w = word2ix[word]
        obs_pos[(w, c)] += 1

        t = tag2ix[prevt]
        tag_pos[(t, c)] += 1


    for (prevt, word), tag in pred_feats:
        c = tag2ix[tag]
        w = word2ix[word]
        obs_pos[(w, c)] += -1

        t = tag2ix[prevt]
        tag_pos[(t, c)] += -1

    return obs_pos, tag_pos


def test_step(gold_feats, pred_feats):
    acc, total = 0, 0
    for ix, v in enumerate(gold_feats):
        total += 1
        if v == pred_feats[ix]:
            acc += 1

    return acc, total


def train_loop(train_data, word2ix, tag2ix, ix2tag, _iwObs, _iwTags, test_data):
    print "Running main training loop ..."
    wObs, wTags = np.copy(_iwObs), np.copy(_iwTags)

    counter = 0
    for wdseq, tgseq in tqdm(train_data):
        counter += 1
        if counter % 30 == 0:
            test_acc, test_total = 0, 0
            for te_wd, te_tg in tqdm(test_data):
                _g, _p = forward(te_wd, te_tg, word2ix, ix2tag, wObs, wTags)
                acc, tot = test_step(_g, _p)
                test_acc += acc
                test_total += tot

            tqdm.write("Completed %d samples. Accuracy: %f"%\
                        (counter, test_acc*1.0/test_total))

        tr_gold, tr_pred = forward(wdseq, tgseq, word2ix, ix2tag, wObs, wTags)
        obs_pos, tag_pos = train_step(tr_gold, tr_pred, word2ix, tag2ix)

        for (r, c), v in obs_pos.items():
            if v != 0:
                wObs[r, c] += v

        for (r, c), v in tag_pos.items():
            if v != 0:
                wTags[r, c] += v

    return wObs, wTags



@plac.annotations(
    path_brown_corpus = ("Path to Brown Corpus dir.", 'positional', None, str)
)
def main(path_brown_corpus):
    """
    Runs Collins' averaged perceptron tagging algorithm.
    """
    print "Files dir:", path_brown_corpus

    if not os.path.isdir(path_brown_corpus):
        raise OSError("path_brown_corpus -- %s does not exist."%path_brown_corpus)

    files = [os.path.join(path_brown_corpus, name) for name in 
             os.listdir(path_brown_corpus) if len(name) == 4 and name[0] == 'c']

    data = load_files(files[:1])
    random.shuffle(data)

    tsplit = int(0.4 * len(data))
    train_data, test_data = data[:tsplit], data[tsplit:]
    print "Train seqs:", len(train_data)
    print "Test  seqs:", len(test_data)

    train_feats = [get_features(w, t) for w,t in tqdm(train_data)]

    #ft2ix, ix2ft, tag2ix, ix2tag, weights  = get_feature_map_and_weights(train_feats)
    word2ix, ix2word, tag2ix, ix2tag, wObs, wTags = get_words_tags_weights(train_feats)

    ipdb.set_trace()
    train_loop(train_data, word2ix, tag2ix, ix2tag, wObs, wTags, train_data)


if __name__ == '__main__':
    plac.call(main)
