# CMake generated Testfile for 
# Source directory: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/mcc
# Build directory: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/mcc
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_mcc "/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/bin/opencv_test_mcc" "--gtest_output=xml:opencv_test_mcc.xml")
set_tests_properties(opencv_test_mcc PROPERTIES  LABELS "Extra;opencv_mcc;Accuracy" WORKING_DIRECTORY "/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/cmake/OpenCVUtils.cmake;1640;add_test;/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/cmake/OpenCVModule.cmake;1310;ocv_add_test_from_target;/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/cmake/OpenCVModule.cmake;1074;ocv_add_accuracy_tests;/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/mcc/CMakeLists.txt;2;ocv_define_module;/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/mcc/CMakeLists.txt;0;")
