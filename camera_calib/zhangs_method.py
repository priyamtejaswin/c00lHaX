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

Each corner location in pixel coordinate with the origin at the top-left.
This follows the image format -- origin for image is at top-left.
If you read a point as `x, y = point`, on an image, `x` will be the column
and `y` will be the row.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        for p in square:
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


img = cv2.imread('/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/camera_calib/data/zhang_data/CalibIm1.tif', -1)
points = load_points('/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/camera_calib/data/zhang_data/data1.txt')

print points[0]
print points[-1]

draw_points([points[0], points[7], points[-1]], img)