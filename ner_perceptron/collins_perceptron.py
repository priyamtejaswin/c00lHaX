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

"""

