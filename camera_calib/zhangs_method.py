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

[3:30 PM] Success! Intrinsic solver values close to answer.
Diff because:
* Radial distortion.
* This is the first closed form solution -- optimisation procedure is pending.

Current matrix is:
```
[[888.129   0.359 278.366]
 [  0.    885.091 197.62 ]
 [  0.      0.      1.   ]]
```
And the answer is:
```
[[832.5    0.204  303.959]
 [  0.    832.53  206.585]
 [  0.      0.      1.   ]]
```

[6th June, 4:46 PM] Extrinsics do not match.
I have doubts about the scale/size of the real-world coordinates.
My guess is that the intrinsics shuold not change -- the extrinsics should adapt.
So my guess is I have not initialised the IRW points correctly.

I tried checking OpenCV, but it only supports a very specific type of calibration,
using the chesssboard. 

Finally figured out what the `Model.txt` file is for -- it contains the IRW data.

[5:19 PM] It's working now.
Had to use the coordinates they had generated.

[6:15 PM] Can't figure out why this data (`Model.txt`) is special.
I was right about the origin being on top left, except the points are reversed in order
-- my first point starts from the bottom, their's starts from the top.
The diff b/w points is 0.5 -- again, this is why I was seeing close numbers for `size=1`.
The mean/std of the data is not meaningful -- for now I believe that the distance doesn't really
matter, unless you are replicating/recreating someone's results.

[7:44 PM] Including radial distortion.
The paper says the alternating optimisation takes a while to converge.
I'm trying the full least-squares version here.

[7th June, 12:15 AM] Spent the rest of the day figuring out Scipy Optimise
and understanding how to model Radial Distortion.
I got the optimiser to work. But it's not working ...

If I give an initial solution with `k1, k2 = 0, 0` the algorthim just returns
the initial solution. Any value slightly higher than 0 gives wild results.
I'm going to guess that my distortion calculation is wrong.

The Zhang MSR paper is not particularly insightful -- I'll try reading the
ICCV paper that was published first.

[8th June, 2:00 AM] Some success...
I've given up at modeling distortion at this point.
Went back and figured out how to do the optimisation properly *without* k1,k2.
The main issue with using an unbounded optimiser (LM in Minpack) was that the
rotation matrix has to be orthonormal.
The Rodrigues formula helps out with this.
Instead of given the optimiser a matrix of vectors, you give it 3 angles.
This is how rotation matrices are expressed in Euler form. Rodrigues formula
converts this euler/angle vector to rotation matrix.

Applying this only only the intrisic+extrinsic matrix (sans distortion) works.
Avg drift was at around 1.044
This reduced to 0.938, and the params were closer to the true values.

With this `breakthrough`, I'll try the distortion model once again.
Peace.
"""


import os
import cv2
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from itertools import izip
from sklearn.preprocessing import StandardScaler


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

    assert i == 3*len(m)
    assert A.shape == (3*len(m), 9)
    u, s, vh = np.linalg.svd(A)
    return vh[np.argmin(s)], A


def zhang_homography(m, M):
    """
    Following the method in Appendix A.
    """
    assert len(m) == len(M), 'Number of points is not the same.'

    assert isinstance(m[0], list) or isinstance(m[0], tuple), '`m` should be list of 2d points.'
    assert len(m[0]) == 2, '`m` should be list of 2d points.'

    assert isinstance(M[0], list) or isinstance(M[0], tuple), '`M` should be list of 3d points.'
    assert len(M[0]) == 3, '`M` should be list of 3d points.'

    A = np.zeros((2 * len(m), 9))

    i = 0
    for image, world in izip(m, M):
        x, y = image
        vec = np.array(world)
        
        A[i, 0:3] = vec
        A[i, 3:6] = 0
        A[i, 6:9] = -1 * x * vec

        A[i+1, 0:3] = 0
        A[i+1, 3:6] = vec
        A[i+1, 6:9] = -1 * y * vec

        i += 2

    assert i == 2*len(m)
    assert A.shape == (2*len(m), 9)
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


def solve_extrinsics(A, H):
    """
    Compute Extrinsic matrix (R, t) for every homography/image.
    Last step following Appendix C.
    """
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    invA = np.linalg.inv(A)
    clam = 1.0 / np.linalg.norm(np.dot(invA, h1))

    r1 = clam * np.dot(invA, h1)
    r2 = clam * np.dot(invA, h2)
    r3 = np.cross(r1, r2)
    t = clam * np.dot(invA, h3)

    Q = np.vstack([r1, r2, r3]).T
    u, s, vh = np.linalg.svd(Q)
    R = np.dot(u, vh)

    return R, t


def normalise_points(points):
    data = np.array(points)
    std = StandardScaler()
    normed = std.fit_transform(data)
    return normed.tolist()


def projection(params, pixels, M, num_images, num_points):
    """
    Compute reprojection error across all pixels, IRW points M and the unknown parameters.
    `params` format:
    [a, g, b, u0, v0] + [k1, k2] + EulerAngle_vector + t.flatten()

    `num_images` is total images in data.
    `num_points` is number of points in every image.
    """
    assert len(params) == 7 + 6*num_images
    assert len(pixels) == num_images*num_points
    assert len(M) == num_points

    A = np.zeros((3, 3))
    A[0, 0], A[0, 1] = params[0], params[1]
    A[1, 1] = params[2]
    A[0, 2], A[1, 2] = params[3], params[4]
    A[2, 2] = 1.0

    # u0, v0 = A[0, 2], A[1, 2]
    k1, k2 = params[5], params[6]

    img_rotations = []
    img_translations = []

    i = 7
    for _ in range(num_images):
        rodvec = np.array(params[i:i+3])
        R = cv2.Rodrigues(rodvec)[0]
        # R = np.array(params[i:i+9]).reshape(3, 3)
        img_rotations.append(R)

        t = np.array(params[i+3:i+6])
        img_translations.append(t)

        i += 6

    residual = []
    for i in range(num_images):
        m = np.array(pixels[i*num_points : (i+1)*num_points])
        R = img_rotations[i]
        t = img_translations[i]

        r1, r2 = R[:, 0], R[:, 1]
        E = np.vstack([r1, r2, t]).T

        # Uncomment next block if you are discounting distortion.
        # ideal = np.dot(A, np.dot(E, M.T)).T
        # pred = np.zeros_like(m)
        # pred[:, 0] = ideal[:, 0] / ideal[:, 2]
        # pred[:, 1] = ideal[:, 1] / ideal[:, 2]

        image_coords = np.dot(E, M.T).T
        normed_coords = np.zeros_like(image_coords)
        assert normed_coords.shape == (M.shape[0], 3)
        normed_coords[:, 0] = image_coords[:, 0] / image_coords[:, 2]
        normed_coords[:, 1] = image_coords[:, 1] / image_coords[:, 2]
        normed_coords[:, 2] = 1.0

        rconst = np.square(normed_coords[:, 0]) + np.square(normed_coords[:, 1])
        r2const = np.square(rconst)
        xcap = np.multiply(normed_coords[:, 0], 1 + k1*rconst + k2*r2const)
        ycap = np.multiply(normed_coords[:, 1], 1 + k1*rconst + k2*r2const)

        dist_coords = np.zeros_like(normed_coords)
        dist_coords[:, 0] = xcap
        dist_coords[:, 1] = ycap
        dist_coords[:, 2] = 1.0

        ideal = np.dot(A, dist_coords.T).T
        pred = np.zeros_like(m)
        pred[:, 0] = ideal[:, 0] / ideal[:, 2]
        pred[:, 1] = ideal[:, 1] / ideal[:, 2]

        error = np.linalg.norm(pred-m, axis=1)
        assert len(error) == len(m)
        residual.extend(error.tolist())

    return np.array(residual)


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

    root_path = '/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/camera_calib/data/zhang_data'
    point_paths = [os.path.join(root_path, 'data%d.txt'%i) for i in range(1, 6)]

    all_points = []
    for f in point_paths:
        array = load_points(f)
        all_points.append([p for s in array for p in s])

    print "total images:", len(all_points)
    print "total points:", sum(len(a) for a in all_points)

    # all_reals = gen_realworld_points(17, 8, 8)
    all_reals = load_points('/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/camera_calib/data/zhang_data/Model.txt')
    M = [(p[0], p[1], 1) for s in all_reals for p in s]
    print "total reals:", len(M)
    assert len(all_points[0]) == len(M)

    hvec, A = solve_homography(all_points[0], M)

    homographies = []
    zhang_homogs = []

    for m in all_points:
        hvec, A = solve_homography(m, M)
        homographies.append(hvec.reshape(3, 3))

        zvec, _ = zhang_homography(m, M)
        zhang_homogs.append(zvec.reshape(3, 3))

    intrinsic_mat = solve_intrinsics(homographies)
    print round(intrinsic_mat[0, 0], 5), round(intrinsic_mat[0, 1], 5), round(intrinsic_mat[1, 1], 5),\
            round(intrinsic_mat[0, 2], 5), round(intrinsic_mat[1, 2], 5)
    print

    zinst_mat = solve_intrinsics(zhang_homogs)

    #[a, g, b, u0, v0] + [k1, k2] + R.flatten() + t.flatten() 
    params = [intrinsic_mat[0, 0], intrinsic_mat[0, 1], intrinsic_mat[1, 1]]
    params += [intrinsic_mat[0, 2], intrinsic_mat[1, 2]]  # u0, v0
    params += [0, 0]  # Distortion k1, k2
    for ix, H in enumerate(homographies):
        r, t = solve_extrinsics(intrinsic_mat, H)
        rodvec = cv2.Rodrigues(r)[0]
        # params += r.flatten().tolist()
        params += rodvec.flatten().tolist()
        params += t.tolist()

        for row in np.vstack([r, t]):
            print ' '.join(map(lambda x:str(round(x, 5)), [_ for _ in row]))
        print

        # Computing projection error.
        m = np.array(all_points[ix])
        rigid = np.vstack([r[:, 0], r[:, 1], t]).T
        proj = np.dot(intrinsic_mat, np.dot(rigid, np.array(M).T)).T
        pred = np.zeros((len(proj), 2))
        pred[:, 0] = proj[:, 0]/proj[:, 2]
        pred[:, 1] = proj[:, 1]/proj[:, 2]

        assert m.shape == pred.shape
        _img_error = np.mean(np.linalg.norm(pred-m, axis=1))
        print "mean error:", _img_error

        _rand_indices = np.random.randint(0, 256, 50)
        crosses = m[_rand_indices]
        plusses = pred[_rand_indices]
        # plt.scatter([_[0] for _ in crosses], [_[1] for _ in crosses], label='truth', marker='x')
        # plt.scatter([_[0] for _ in plusses], [_[1] for _ in plusses], label='preds', marker='+')
        # plt.title('Mean pixel deviation for image: %f'%round(_img_error, 2))
        # plt.legend()
        # plt.show()

    # Trying the Optimizer ... Needs more work.
    assert len(params) == 7 + 6*len(point_paths)

    pixels = [p for l in all_points for p in l]
    print "initial error:", np.mean(projection(np.array(params), np.array(pixels), np.array(M),
                                        num_images=len(point_paths), num_points=len(M)))

    print "Running optimiser ..."
    opresult = scipy.optimize.least_squares(
        fun=projection, x0=np.array(params), 
        args=(np.array(pixels), np.array(M), len(point_paths), len(M)),
        method='lm'
    )
    print "final error:", np.mean(projection(np.array(opresult.x), np.array(pixels), np.array(M),
                                        num_images=len(point_paths), num_points=len(M)))

    # Plotting the last image points ...
    postop_intrinsic = np.zeros((3, 3))
    postop_intrinsic[0, 0] = opresult.x[0]
    postop_intrinsic[1, 1] = opresult.x[2]
    postop_intrinsic[0, 1] = opresult.x[1]
    postop_intrinsic[0, 2] = opresult.x[3]
    postop_intrinsic[1, 2] = opresult.x[4]
    postop_intrinsic[2, 2] = 1.0
    dk1, dk2 = opresult.x[5], opresult.x[6]
    euler_vec = opresult.x[-6:-3]
    trans_vec = opresult.x[-3:]

    R_mat = cv2.Rodrigues(euler_vec)[0]
    r1, r2 = R_mat[:, 0], R_mat[:, 1]
    E = np.vstack([r1, r2, trans_vec]).T

    image_coords = np.dot(E, np.array(M).T).T
    normed_coords = np.zeros_like(image_coords)
    assert normed_coords.shape == (len(M), 3)
    normed_coords[:, 0] = image_coords[:, 0] / image_coords[:, 2]
    normed_coords[:, 1] = image_coords[:, 1] / image_coords[:, 2]
    normed_coords[:, 2] = 1.0

    rconst = np.square(normed_coords[:, 0]) + np.square(normed_coords[:, 1])
    r2const = np.square(rconst)
    xcap = np.multiply(normed_coords[:, 0], 1 + dk1*rconst + dk2*r2const)
    ycap = np.multiply(normed_coords[:, 1], 1 + dk1*rconst + dk2*r2const)

    dist_coords = np.zeros_like(normed_coords)
    dist_coords[:, 0] = xcap
    dist_coords[:, 1] = ycap
    dist_coords[:, 2] = 1.0

    ideal = np.dot(postop_intrinsic, dist_coords.T).T
    pred = np.zeros_like(m)
    pred[:, 0] = ideal[:, 0] / ideal[:, 2]
    pred[:, 1] = ideal[:, 1] / ideal[:, 2]

    # _rand_indices = np.random.randint(0, 256, 50)
    # crosses = m[_rand_indices]
    # plusses = pred[_rand_indices]
    plt.scatter([_[0] for _ in crosses], [_[1] for _ in crosses], label='truth', marker='x')
    plt.scatter([_[0] for _ in plusses], [_[1] for _ in plusses], label='closed soln, sans rdist', marker='+')
    circles = pred[_rand_indices]
    plt.scatter([_[0] for _ in circles], [_[1] for _ in circles], label='postop soln', s=5)
    plt.legend()
    plt.show()


    # Printing final parameters ...
    _iarray = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    towrite = []
    _a = [postop_intrinsic[a, b] for a,b in _iarray]
    print _a
    towrite.append(_a)
    print
    print dk1, dk2
    towrite.append([dk1, dk2])
    print
    i = 7
    for _ in range(len(point_paths)):
        euler_vec = opresult.x[i:i+3]
        trans_vec = opresult.x[i+3:i+6]
        i += 6

        R_mat = cv2.Rodrigues(euler_vec)[0]
        print R_mat.T
        towrite.extend(R_mat.tolist())
        print trans_vec
        towrite.append(trans_vec)
        print

    import csv
    with open('results.txt', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(towrite)