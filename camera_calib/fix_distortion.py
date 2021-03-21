#!/usr/bin/env python
"""
created at: Sun Jun 14 14:39:22 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Fix distortion. Pass the image number and image path.
"""


import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt


rfile = sys.argv[1]
im_file = sys.argv[2]
im_ix = int(sys.argv[3])

intrinsic = np.zeros((3, 3))
extrinsics = []
rdist = [None, None]

with open(rfile) as fp:
    first = map(float, fp.readline().strip().split(','))
    intrinsic[0, 0] = first[0]
    intrinsic[1, 1] = first[2]
    intrinsic[0, 1] = first[1]
    intrinsic[0, 2] = first[3]
    intrinsic[1, 2] = first[4]
    intrinsic[2, 2] = 1.0

    second = map(float, fp.readline().strip().split(','))
    rdist[0] = second[0]
    rdist[1] = second[1]

    for i in range(5):
        m = []
        for _ in range(3):
            row = map(float, fp.readline().strip().split(','))
            m.append(row)

        rmat = np.array(m)
        tvec = np.array(map(float, fp.readline().strip().split(',')))
        extrinsics.append(
            np.vstack([rmat[:, 0], rmat[:, 1], tvec]).T
        )

img = cv2.imread(im_file, -1)
plt.imshow(img)
plt.show()
