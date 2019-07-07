# Notes on Trees. And stuff.
...In VIM. All of it. Oh yeah. Deal with it.

*Only took me 2 hours to begin using screen buffers with a new keymap for splits.*

*Perhaps this was not such a good idea...*

Main reference is [the ESL book](https://web.stanford.edu/~hastie/ElemStatLearn/).

## Introduction.
Two main algorithms: CART and C4.5. A regression tree splits the feature-space into "rectangles" -- you pick a feature at random, take a cut at random, and the data is split on that cut.

In 2d feature-space, these are actual rectangles.

`TODO: Add an image here`

The **response** from each region is the average of all the target values $y_i$ in that region.

How to find the best partition? Consider a variable $j$ and a split point $s$. With these selected, we can define two regions $R_1, R_2$ such that

$$
R_1(j, s) = {X|X_j < s}\ \text{and} \ R_2(j, s) = {X|X_j > s}
$$

We want to find these regions such that the overall Sum of Squared Errors (SSE) in the data is minimal
$$
\sum_{r \in R} \sum_{i \in r} (y_i - \hat{y_r})^2
$$

Algorithm:
1. Take all the data into one collection and calculate the SSE.
2. Iterate over all variables and the values per variable to find possible split points. Select the best one which minimizes the SSE compared to the parent SSE.
3. Partition the data into two regions based on the split.
4. Repeat from (2) for the two new regions until convergence.

The convergence criterion could be a minimum sample size per leaf or that the reduction in SSE has to exceed some threshold.

## Pruning.
In practice, we avoid the threshold convergence -- a better, more worthwhile split could come after a poor split. We rather let the tree get built with the min-samples criteria and then apply pruning.

With the data points at the leaf nodes, a pruned tree is simply a sub-tree $T_c$ with some intermediate nodes missing from the master tree $T_0$. We will *try* different sub-trees and pick the one which minimizes the test-error. Of course, trying all possible combinations ($2^n$ for $n$ intermediate nodes) is not an option.


