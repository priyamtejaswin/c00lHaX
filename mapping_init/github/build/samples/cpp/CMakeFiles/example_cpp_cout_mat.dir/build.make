# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.18.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.18.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build

# Include any dependencies generated for this target.
include samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/depend.make

# Include the progress variables for this target.
include samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/progress.make

# Include the compile flags for this target's objects.
include samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/flags.make

samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.o: samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/flags.make
samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/samples/cpp/cout_mat.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/samples/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/samples/cpp/cout_mat.cpp

samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/samples/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/samples/cpp/cout_mat.cpp > CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.i

samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/samples/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/samples/cpp/cout_mat.cpp -o CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.s

# Object files for target example_cpp_cout_mat
example_cpp_cout_mat_OBJECTS = \
"CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.o"

# External object files for target example_cpp_cout_mat
example_cpp_cout_mat_EXTERNAL_OBJECTS =

bin/example_cpp_cout_mat: samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/cout_mat.cpp.o
bin/example_cpp_cout_mat: samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/build.make
bin/example_cpp_cout_mat: 3rdparty/lib/libippiw.a
bin/example_cpp_cout_mat: 3rdparty/ippicv/ippicv_mac/icv/lib/intel64/libippicv.a
bin/example_cpp_cout_mat: lib/libopencv_gapi.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_stitching.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_alphamat.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_aruco.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_bgsegm.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_bioinspired.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_ccalib.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_dnn_objdetect.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_dnn_superres.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_dpm.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_face.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_freetype.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_fuzzy.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_hdf.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_hfs.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_img_hash.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_intensity_transform.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_line_descriptor.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_mcc.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_quality.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_rapid.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_reg.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_rgbd.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_saliency.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_sfm.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_stereo.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_structured_light.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_superres.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_surface_matching.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_tracking.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_videostab.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_xfeatures2d.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_xobjdetect.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_xphoto.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_highgui.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_shape.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_datasets.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_plot.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_text.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_ml.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_dnn.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_phase_unwrapping.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_optflow.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_ximgproc.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_videoio.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_video.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_imgcodecs.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_objdetect.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_calib3d.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_features2d.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_flann.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_photo.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_imgproc.4.5.0.dylib
bin/example_cpp_cout_mat: lib/libopencv_core.4.5.0.dylib
bin/example_cpp_cout_mat: samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/example_cpp_cout_mat"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/samples/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_cpp_cout_mat.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/build: bin/example_cpp_cout_mat

.PHONY : samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/build

samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/samples/cpp && $(CMAKE_COMMAND) -P CMakeFiles/example_cpp_cout_mat.dir/cmake_clean.cmake
.PHONY : samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/clean

samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/samples/cpp /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/samples/cpp /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : samples/cpp/CMakeFiles/example_cpp_cout_mat.dir/depend

