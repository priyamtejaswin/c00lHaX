#!/usr/bin/env python
"""
created at: Sun Aug 25 14:41:52 2019
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

[Collins, "Average Perceptron", EMNLP 2002](https://www.aclweb.org/anthology/W02-1001)

Extended the general perceptron algorithm to `structured` predition:
1. Assign weights
2. Run Viterbi Inference
3. Compare predicted tags against truth.
4. Update weights.

Paper seems *simple enough*.
"""

