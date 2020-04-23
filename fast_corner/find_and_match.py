#!/usr/bin/env python
"""
created at: Thu Apr 23 13:58:44 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Given two images from different perspectives, this script
applied FAST + BRIEF to match keypoints.

Usage: `python find_and_match.py -h`
"""


import os
import cv2
import plac
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


np.random.seed(1)
random.seed(1)


sites = []
for a in np.arange(-23, 24):
    for b in np.arange(-23, 24):
        sites.append((a, b))

p1 = random.sample(sites, 512)
p2 = random.sample(sites, 512)

print len(set(p1)), len(set(p2)), len(set(zip(p1, p2)))

def get_fast_corners(image, threshold=26):
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(threshold)
    fast.setNonmaxSuppression(1)

    kp = fast.detect(image, None)
    markers = cv2.drawKeypoints(image, kp, None, color=(255, 0, 0))

    return kp, markers


def get_brief_descriptors(image, keypoints, pseq):
    descriptor = {}
    for val in keypoints:
        pixel = (int(val.pt[0]), int(val.pt[1]))
        vector = []

        try:
            for a, b in pseq:
                x = image[a[1] + pixel[1], a[0] + pixel[0]]
                y = image[b[1] + pixel[1], b[0] + pixel[0]]
                vector.append(x < y)

            descriptor[pixel] = vector

        except Exception as error:
            print "Failed for", pixel, error

    return descriptor


def matcher(d1, d2, threshold=0.8):
    final = []
    keys = d2.keys()
    values = np.array([d2[k] for k in keys])

    for k,v in d1.items():
        score = np.mean(values == v, axis=1)
        match = np.argmax(score)
        if score[match]>=threshold:
            final.append((k, keys[match]))

    return final


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    assert rows >= 300, "Rows less than 300; get a higher-res photo."
    assert cols >= 300, "Cols less than 300; get a higher-res photo."

    fx = 300.0 / rows
    fy = 300.0 / cols

    smaller = cv2.resize(img, (0, 0), fx=fx, fy=fy)

    blur = cv2.GaussianBlur(smaller, (5, 5), 0)

    return blur


@plac.annotations(left="First image absolute path",
                  right="Second image absolute path")
def main(left, right):
    """
    Accepts two images -- left, right views of the same object.
    Detects FAST corners.
    Computes BRIEF descriptors.
    Plots matching features.
    """

    assert os.path.isabs(left), "Please provide the absolute path; it's a CV2 thing."
    assert os.path.isabs(right), "Please provide the absolute path; it's a CV2 thing."

    i1 = load_image(left)
    i2 = load_image(right)

    kps1, corners1 = get_fast_corners(i1)
    plt.figure(figsize=(10, 10))
    plt.imshow(corners1)
    plt.show()

    kps2, corners2 = get_fast_corners(i2)
    plt.figure(figsize=(10, 10))
    plt.imshow(corners2)
    plt.show()

    bdescr1 = get_brief_descriptors(i1, kps1, zip(p1, p2))
    bdescr2 = get_brief_descriptors(i2, kps2, zip(p1, p2))

    # Plot matching descriptors.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(corners1)
    ax2.imshow(corners2)

    for one, two in matcher(bdescr1, bdescr2, threshold=0.82):
        conpatch = patches.ConnectionPatch(xyA=two, xyB=one, coordsA="data", coordsB="data", color='green',
                                          axesA=ax2, axesB=ax1)
        ax2.add_artist(conpatch)
    plt.show()


if __name__ == '__main__':
    plac.call(main)
