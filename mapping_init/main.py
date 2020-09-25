#!/usr/bin/env python
"""
created at: Wed Sep 23 20:31:17 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Main file to start map initialization.
Following PTAM 2007
<http://www.robots.ox.ac.uk/~gk/publications/KleinMurray2007ISMAR.pdf>
"""


import numpy as np
import cv2 as cv
import os


def get_camera_feed():
    """
    Return some camera object.
    """
    return


def load_img(img_path, resize=True):
    raw = cv.imread(img_path, 0)
    if resize:
        height, width = raw.shape
        if height >= 500 and height <= 600:
            return raw
        elif height > 600:
            scale = round(height/550.0, 2)
            res_w = int(round(width/scale))
            # resize(src, (new_width, new_height)
            return cv.resize(raw, (res_w, 550))
        else:
            raise ValueError("Input shape (%d, %d) is too small.", rows, cols)


def get_first_keyframe():
    """
    Write something on screen -- tell user to hit <space> to capture.
    """
    # Sample `right` image for now ...
    return load_img('./samples/aloeR.jpg')


def get_second_keyfame():
    """
    Write something on screen -- tell user to hit <space> to capture.
    """
    # Sample 'left' image for now ...
    return load_img('./samples/aloeL.jpg')


def eight_pt(pts_before, pts_after, hnorm, wnorm):
    assert pts_before.shape == pts_after.shape
    size = pts_before.shape[0]
    assert size >= 8

    A = np.zeros((9 * size, 9))

    for ix in range(size):
        y1, y2 = pts_before[ix]  # Before (y)
        y1, y2 = hnorm(y1), wnorm(y2)

        yp1, yp2 = pts_after[ix]  # After (y')
        yp1, yp2 = hnorm(yp1), wnorm(yp2)

        row = [yp1 * y1, yp1 * y2, yp1,
               yp2 * y1, yp2 * y2, yp2,
               y1, y2, 1]

        A[ix] = row

    u, s, vh = np.linalg.svd(A)
    Eest = vh[np.argmin(s)].reshape(3, 3)

    # Enforce the singular value consistency in the Essential Matrix.
    u, s, vh = np.linalg.svd(Eest)
    s.sort()
    sprime = np.zeros((3, 3))
    sprime[0, 0] = s[-1]
    sprime[1, 1] = s[-2]

    return u.dot(sprime).dot(vh)


def prepare_matches(kps_first, kps_second, matches):
    points_first, points_second = [], []
    for m in matches:
        if m.imgIdx == 0:
            train, query = kps_first[m.trainIdx].pt, kps_second[m.queryIdx].pt
        elif m.imgIdx == 1:
            query, train = kps_first[m.trainIdx].pt, kps_second[m.queryIdx].pt
        else:
            raise ValueError("Unknown imgIdx in DMatch object.")

        points_first.append(train)
        points_second.append(query)

    return np.array(points_first), np.array(points_second)


def znm(mu, sigma):
    def normalize(x, m=mu, s=sigma):
        return (x - m)/(1.0 * s)
    return normalize


def main():
    kf_first = get_first_keyframe()
    kf_second = get_second_keyfame()

    print kf_first.shape, kf_second.shape
    assert kf_first.shape == kf_second.shape, "1st, 2nd keyframes shapes mismatch"

    height, width = kf_first.shape
    hmu, hsig = np.mean(np.arange(height)), np.std(np.arange(height))
    wmu, wsig = np.mean(np.arange(width)), np.std(np.arange(width))

    hnorm = znm(hmu, hsig)
    wnorm = znm(wmu, wsig)

    orb = cv.ORB_create()
    bfm = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    kps_first, des_first = orb.detectAndCompute(kf_first, None)
    kps_second, des_second = orb.detectAndCompute(kf_second, None)

    matches = bfm.match(des_first, des_second)
    total = len(matches)
    top50 = int(0.5 * total)
    top_matches = sorted(matches, key=lambda x:x.distance)[:top50]

    pts_first, pts_second = prepare_matches(kps_first, kps_second, top_matches[:8])

    Emat = eight_pt(pts_before=pts_first, pts_after=pts_second, hnorm=hnorm, wnorm=wnorm)
    print Emat


if __name__ == '__main__':
    print "Starting ..."
    main()
