#!/usr/bin/env python
"""
created at: Wed Oct 21 14:25:11 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Trying OpenCV with Boost and Python3.
"""


from distutils.core import setup
from distutils.extension import Extension

krtfromp = Extension(
    'krtfromp',
    sources=['krtfromp.cpp'],
    include_dirs = ['/usr/local/include'],
    library_dirs = ['/usr/local/lib'],
    libraries=['boost_python38', 'opencv_core', 'opencv_highgui', 'opencv_sfm'],
)

setup(
    name='opencv-krt',
    version='0.1',
    ext_modules=[krtfromp])

