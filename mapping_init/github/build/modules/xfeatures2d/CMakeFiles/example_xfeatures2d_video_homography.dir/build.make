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
include modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/depend.make

# Include the progress variables for this target.
include modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/progress.make

# Include the compile flags for this target's objects.
include modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/flags.make

modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.o: modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/flags.make
modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/xfeatures2d/samples/video_homography.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/xfeatures2d && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/xfeatures2d/samples/video_homography.cpp

modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/xfeatures2d && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/xfeatures2d/samples/video_homography.cpp > CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.i

modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/xfeatures2d && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/xfeatures2d/samples/video_homography.cpp -o CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.s

# Object files for target example_xfeatures2d_video_homography
example_xfeatures2d_video_homography_OBJECTS = \
"CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.o"

# External object files for target example_xfeatures2d_video_homography
example_xfeatures2d_video_homography_EXTERNAL_OBJECTS =

bin/example_xfeatures2d_video_homography: modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/samples/video_homography.cpp.o
bin/example_xfeatures2d_video_homography: modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/build.make
bin/example_xfeatures2d_video_homography: lib/libopencv_xfeatures2d.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_ml.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_shape.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_highgui.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_calib3d.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_features2d.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_flann.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_videoio.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_imgcodecs.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_imgproc.4.5.0.dylib
bin/example_xfeatures2d_video_homography: lib/libopencv_core.4.5.0.dylib
bin/example_xfeatures2d_video_homography: modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/example_xfeatures2d_video_homography"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/xfeatures2d && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_xfeatures2d_video_homography.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/build: bin/example_xfeatures2d_video_homography

.PHONY : modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/build

modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/xfeatures2d && $(CMAKE_COMMAND) -P CMakeFiles/example_xfeatures2d_video_homography.dir/cmake_clean.cmake
.PHONY : modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/clean

modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/xfeatures2d /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/xfeatures2d /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/xfeatures2d/CMakeFiles/example_xfeatures2d_video_homography.dir/depend

