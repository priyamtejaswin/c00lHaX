#!/bin/bash

cmake \
-D CMAKE_BUILD_TYPE=Release \
-D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules \
-D BUILD_EXAMPLES=ON \
-D PYTHON3_EXECUTABLE=/usr/local/bin/python3 \
-D PYTHON3_INCLUDE_DIR=/usr/local/Cellar/python@3.8/3.8.6/Frameworks/Python.framework/Versions/3.8/include/python3.8 \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.8/site-packages/numpy/core/include/
../opencv-master

