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
- Sticking with the POS taggubg task -- unable to find the other dataset.
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

[00:09] - Got distracted. Resuming now.
- Trying to wrap my head around the feature representation.
- A feature here is just a <condition,tag> pair. Global feature is sum of 
counts of said local features. Every global feature has a weight (alpha) with 
it. The training procedure updates these alphas.

"""


import os
import sys
import plac
import random
from tqdm import tqdm


def get_clean_line(line):
    line = line.lower().strip()
    if len(line) < 1:
        return None
    else:
        words, tags = [], []

        for token in line.split():
            word, tag = token.strip().split('/')
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
    _pad_false = lambda s: [None, None] + s + [None, None]

    words = _pad_false(words)
    tags = _pad_false(tags)
    rare = _pad_false(w_is_rare)

    features = []

    for ix in range(2, 2+size):
        if rare[ix] is True:
            pass  # We'll figure this out later.
        else:
            # $h_i = {w_i, w_{i+1}, w_{i+2}, w_{i-1}, w_{i-2}, t_{i-1}, t_{i-2}}
            features.append(((words[ix],
                              words[ix+1], words[ix+2],
                              words[ix-1], words[ix-2],
                              tags[ix-1], tags[ix-2]), tags[ix]))

    return features


def load_files(files_list):
    data = []
    c = 0

    for f in tqdm(files_list):
        with open(f, 'r') as fp:
            for line in fp.readlines():
                tokens = get_clean_line(line)
                if tokens is not None:
                    c += 1
                    data.append(tokens)
    print "Found %d sequences."%c
    return data


def get_feature_map(list_of_seq_feats):
    ft2ix = {}
    ix2ft = []
    c = 0

    for seq in tqdm(list_of_seq_feats):
        for feat in seq:
            if feat not in fmap:
                ft2ix[feat] = c
                ix2ft.append(feat)
                c += 1

    print "Found %d features."%c
    return ft2ix, ix2ft


@plac.annotations(
    path_brown_corpus = ("Path to Brown Corpus file.", 'positional', None, str)
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

    random.shuffle(files)
    tsplit = int(0.8 * len(files))
    train_files, test_files = files[:tsplit], files[tsplit:]
    print "Found %d train files."%len(train_files)
    print "Found %d test files."%len(test_files)



if __name__ == '__main__':
    plac.call(main)
