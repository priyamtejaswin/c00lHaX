#!/usr/bin/env python
"""
created at: Mon Dec 23 15:20:06 2019
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

[Lafferty, McCallum, Pereira, "Conditional Random Fields", ICML 2001](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

A lot of this code (I/O, utilities, decoding) will be picked from
`../ner_perceptron/collins_perceptron.py`

### 25th Dec
[13:35] - Reading the CRF paper again.

[17:09] - Read. Ate. Read some more.

[23:49] - Initial pass through the paper.

### 7th Jan
[10:43] - Setting up data, i/o and boilerplate stuff.

### 21st Jan
[17:00] - 
"""


import os
import sys
import plac
import ipdb
import time
import random
random.seed(23)
import array
import numpy as np
from tqdm import tqdm
from collections import defaultdict


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


def get_features(seq, tags):


@plac.annotations(
    path_brown_corpus = ("Path to Brown Corpus dir.", 'positional', None, str)
)
def main(path_brown_corpus):
    """
    Runs CRF.
    """
    print "Files dir:", path_brown_corpus

    if not os.path.isdir(path_brown_corpus):
        raise OSError("path_brown_corpus -- %s does not exist."%path_brown_corpus)

    files = [os.path.join(path_brown_corpus, name) for name in 
             os.listdir(path_brown_corpus) if len(name) == 4 and name[0] == 'c']

    data = load_files(files[:50])
    random.shuffle(data)

    tsplit = int(0.9 * len(data))
    train_data, test_data = data[:tsplit], data[tsplit:]
    print "Train seqs:", len(train_data)
    print "Test  seqs:", len(test_data)
    ipdb.set_trace()


if __name__ == '__main__':
    plac.call(main)
