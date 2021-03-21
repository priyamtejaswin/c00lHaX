# CMake generated Testfile for 
# Source directory: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/face
# Build directory: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/face
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_face "/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/bin/opencv_test_face" "--gtest_output=xml:opencv_test_face.xml")
set_tests_properties(opencv_test_face PROPERTIES  LABELS "Extra;opencv_face;Accuracy" WORKING_DIRECTORY "/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/cmake/OpenCVUtils.cmake;1640;add_test;/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/cmake/OpenCVModule.cmake;1310;ocv_add_test_from_target;/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/cmake/OpenCVModule.cmake;1074;ocv_add_accuracy_tests;/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/face/CMakeLists.txt;2;ocv_define_module;/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/face/CMakeLists.txt;0;")
