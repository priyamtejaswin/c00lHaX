#!/usr/bin/env python
"""
created at: Sun Sep  8 19:26:08 2019
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

For Cythonizing.
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("viterbi.pyx"),
    include_dirs=[numpy.get_include()]
)
