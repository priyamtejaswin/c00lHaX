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

[9:10 PM] Finished the plotting code.
FYI --  squares in a file are identified from the bottom-left,
going left to right. So the first square in the file is the one of the bottom-left,
the 8th square is on the bottom-right. The last square is on the top-right.

The square corners are captured clock-wise.

Each corner location in pixel coordinate with the origin at the top-left.
This follows the image format -- origin for image is at top-left.
If you read a point as `x, y = point`, on an image, `x` will be the column
and `y` will be the row.

2. Generate the "real-world" point coordinates.
Each point on the square has a 3D position in real-world.
Z will be 0 with the planar assumption.

[4th June, 5:58 PM]
We have corresponding real-world points.

3. Init function to generate parameter placeholders.
[11:45 PM] Added the Homography solver.

4. Intrinsic solver.
Input is the homography matrix.
Exploits the rotation matrix orthonormal constraints.
[5th June, 12:51 AM] Worked out the Intrinsic parameter solution.
It's another linear-system transformation. Check the end of the notes;
will complete this tomorrow morning.

[12:52 AM] Peace out.
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip


def load_points(path):
    points = []
    with open(path) as fp:
        for line in fp.readlines():
            data = map(float, line.strip().split())
            square = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
            points.append(square)

    return points


def draw_points(points, image=None, size=(480, 640)):
    if image is None:
        final = np.zeros(list(size) + [3])
    else:
        if len(image.shape) == 2:
            # Grayscale ==> RGB
            final = np.concatenate([np.expand_dims(image, -1)]*3, -1)
        else:
            final = np.copy(image)

    for c, square in enumerate(points):
        for q, p in enumerate(square):
            x, y = p

            final[int(y), int(x), c%3] = 255
            final[int(y), int(x), (c+1)%3] = 0
            final[int(y), int(x), (c+2)%3] = 0

            for i,j in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                final[int(y)+i, int(x)+j, c%3] = 255
                final[int(y)+i, int(x)+j, (c+1)%3] = 0
                final[int(y)+i, int(x)+j, (c+2)%3] = 0

    plt.imshow(final)
    plt.show()
    return final


def gen_realworld_points(size, rows, cols, ziszero=True):
    """
    WARNING!
    The origin is on the bottom left corner.
    This is for a very specific pattern -- modify if the pattern changes.
    `size` is in real-world distance (10mm, 5cm, etc).
    """
    if ziszero is False:
        raise NotImplementedError("Not implemented for non-zero Z!")
    
    points = []
    
    start_xs = reversed([i*2*size for i in range(rows)])
    start_ys = [i*2*size for i in range(cols)]

    for x in start_xs:
        for y in start_ys:
            square = []
            for q,w in [(0, 0), (0, size), (size, size), (size, 0)]:
                # Remember, the format should be in (column, row).
                square.append((y+w, x+q))

            points.append(square)

    return points


def solve_homography(m, M):
    """
    Solving `Ah = 0`.
    """
    assert len(m) == len(M), 'Number of points is not the same.'

    assert isinstance(m[0], list) or isinstance(m[0], tuple), '`m` should be list of 2d points.'
    assert len(m[0]) == 2, '`m` should be list of 2d points.'

    assert isinstance(M[0], list) or isinstance(M[0], tuple), '`M` should be list of 3d points.'
    assert len(M[0]) == 3, '`M` should be list of 3d points.'

    A = np.zeros((3 * len(m), 9))

    i = 0
    for image, world in izip(m, M):
        x, y = image
        vec = np.array(world)
        
        A[i, 0:3] = 0
        A[i, 3:6] = -1 * vec
        A[i, 6:9] = y * vec

        A[i+1, 0:3] = vec
        A[i+1, 3:6] = 0
        A[i+1, 6:9] = -1 * x * vec

        A[i+2, 0:3] = -1 * y * vec
        A[i+2, 3:6] = x * vec
        A[i+2, 6:9] = 0

        i += 3

    assert A.shape == (3*len(m), 9)
    u, s, vh = np.linalg.svd(A)
    return vh[np.argmin(s)], A


def solve_intrinsics(hmats):
    """
    Given homography matrices (3 x 3), determine intrinsics.
    ```
    v{i}{j} = [ h{i}[0]*h{j}[0],
            h{i}[0]*h{j}[1] + h{i}[1]*h{j}[0],
            h{i}[1]*h{j}[1],
            h{i}[2]*h{j}[0] + h{i}[0]*h{j}[2],
            h{i}[2]*h{j}[1] + h{i}[1]*h{j}[2],
            h{i}[2]*h{j}[2]]
    ```
    """
    capV = []

    for homography in hmats:
        # Column vectors
        h1, h2 = homography[:, 0], homography[:, 1]

        v12 = [ h1[0]*h2[0],
            h1[0]*h2[1] + h1[1]*h2[0],
            h1[1]*h2[1],
            h1[2]*h2[0] + h1[0]*h2[2],
            h1[2]*h2[1] + h1[1]*h2[2],
            h1[2]*h2[2]]

        v11 = [ h1[0]*h1[0],
            h1[0]*h1[1] + h1[1]*h1[0],
            h1[1]*h1[1],
            h1[2]*h1[0] + h1[0]*h1[2],
            h1[2]*h1[1] + h1[1]*h1[2],
            h1[2]*h1[2]]

        v22 = [ h2[0]*h2[0],
            h2[0]*h2[1] + h2[1]*h2[0],
            h2[1]*h2[1],
            h2[2]*h2[0] + h2[0]*h2[2],
            h2[2]*h2[1] + h2[1]*h2[2],
            h2[2]*h2[2]]

        vmat = np.vstack([np.array(v12), np.array(v11) - np.array(v22)])
        capV.append(vmat)
        print vmat.shape

    capV = np.vstack(capV) 

    assert capV.shape == (2*len(hmats), 6)
    u, s, vh = np.linalg.svd(capV)
    bvec = vh[np.argmin(s)]

    b11, b12, b22, b13, b23, b33 = bvec.tolist()

    v0 = (b12*b13 - b11*b23) / (b11*b22 - b12**2)
    clam = b33 - (b13**2 + v0*(b12*b13 - b11*b23))/b11
    alpha = np.sqrt(clam/b11)
    beta = np.sqrt(clam*b11 / (b11*b22 - b12**2))
    gamma = -1 * b12 * alpha**2 * beta / clam
    u0 = gamma*v0/beta - b13 * alpha**2 / clam

    return np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])



if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    img = cv2.imread('/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/camera_calib/data/zhang_data/CalibIm1.tif', -1)
    points = load_points('/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/camera_calib/data/zhang_data/data1.txt')

    # Plots the 1st, 8th, and last squares on the image.
    # draw_points([points[0], points[7], points[-1]], img)

    reals = gen_realworld_points(10, 3, 3)
    # draw_points([reals[0], reals[1], reals[-1]], size=(200, 200))

    m = [p for s in points[:1] for p in s]
    M = [(p[0], p[1], 1) for s in reals[:1] for p in s]

    _, A = solve_homography(m, M)  # Toy data ...
    assert np.allclose(A[0], [0, 0, 0,
                            0, -40, -1,
                            0, 405.5767*40, 405.5767], rtol=1e-5, atol=1e-5)
    assert np.allclose(A[1], [0, 40, 1,
                            0, 0, 0,
                            0, -63.4392*40, -63.4392], rtol=1e-5, atol=1e-5)
    assert np.allclose(A[2], [0, -405.5767*40, -405.5767,
                            0, 63.4392*40, 63.4392,
                            0, 0,  0], rtol=1e-5, atol=1e-5)

    root_path = '/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/camera_calib/data/msr_data'
    point_paths = [os.path.join(root_path, 'ip_%d.txt'%i) for i in range(1, 6)]

    all_points = []
    for f in point_paths:
        array = load_points(f)
        all_points.append([p for s in array for p in s])

    print "total images:", len(all_points)
    print "total points:", sum(len(a) for a in all_points)

    all_reals = gen_realworld_points(1, 8, 8)
    M = [(p[0], p[1], 1) for s in all_reals for p in s]
    print "total reals:", len(M)
    assert len(all_points[0]) == len(M)

    hvec, A = solve_homography(all_points[0], M)
    print hvec
    print A.shape

    homographies = []
    for m in all_points:
        hvec, A = solve_homography(m, M)
        print hvec.reshape(3, 3)
        homographies.append(hvec.reshape(3, 3))

    intrinsic_mat = solve_intrinsics(homographies)
    print intrinsic_mat


    