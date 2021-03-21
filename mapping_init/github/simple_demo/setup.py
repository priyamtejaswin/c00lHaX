#!/usr/bin/env python
"""
created at: Wed Oct 21 14:25:11 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Trying Boost with Python3.
"""


from distutils.core import setup
from distutils.extension import Extension

hello_ext = Extension(
    'hello_ext',
    sources=['hello_ext.cpp'],
    libraries=['boost_python38'],
)

setup(
    name='hello-world',
    version='0.1',
    ext_modules=[hello_ext])

