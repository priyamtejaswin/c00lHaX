#!/usr/bin/env python
"""
created at: Tue Nov 24 22:46:17 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Trying `pptk`
"""


import numpy as np
s = np.linspace(0.0, 2 * np.pi, 1000)[None, :]
t = np.linspace(-1.0, 1.0, 50)[:, None]

x = (1 + 0.5 * t * np.cos(0.5 * s)) * np.cos(s)
y = (1 + 0.5 * t * np.cos(0.5 * s)) * np.sin(s)
z = 0.5 * t * np.sin(0.5 * s)
P = np.stack([x, y, z], axis=-1)


N = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
N /= np.sqrt(np.sum(N ** 2, axis=-1))[:, :, None]

import pptk
v = pptk.viewer(P)
v.attributes(0.5 * (N.reshape(-1, 3) + 1))
v.set(point_size=0.001)

