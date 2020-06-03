#!/usr/bin/env python
"""
created at: Wed Jun  3 13:30:38 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Took a lot of notes -- here we go.

<https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf>

Following Wikipedia, I have some idea of how to solve for the Homography.
After reading Zhang's appendix, there's a chance my formulation is slightly off.
Or at least, sub-optimal -- the last equation that I include is redundant.

Anyways.
Here's the breakdown.

1. Load/plot image data.
You would typically run a corner detection algorithm on these images,
but I'm going to read the text files provided by Zhang with the paper.
I've read different approaches to do this -- they are indeed specifically
designed for corner detection. The algorithm that OpenCV uses is disucssed here
-- [Automatic Detection of Checkerboards](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/14476/eth-30969-01.pdf)
The Zhang paper fits lines to edges of individual squares, and then considers
the intersection as the corners. A reasonable approach imo. Fish-eye will be
minimal for small squares.


"""


import cv2
import numpy as np
import matplotlib.pyplot as plt



