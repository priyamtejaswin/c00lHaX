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
include modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/depend.make

# Include the progress variables for this target.
include modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/progress.make

# Include the compile flags for this target's objects.
include modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/flags.make

modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.o: modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/flags.make
modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/face/samples/sampleDetectLandmarksvideo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/face && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/face/samples/sampleDetectLandmarksvideo.cpp

modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/face && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/face/samples/sampleDetectLandmarksvideo.cpp > CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.i

modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/face && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/face/samples/sampleDetectLandmarksvideo.cpp -o CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.s

# Object files for target example_face_sampleDetectLandmarksvideo
example_face_sampleDetectLandmarksvideo_OBJECTS = \
"CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.o"

# External object files for target example_face_sampleDetectLandmarksvideo
example_face_sampleDetectLandmarksvideo_EXTERNAL_OBJECTS =

bin/example_face_sampleDetectLandmarksvideo: modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/samples/sampleDetectLandmarksvideo.cpp.o
bin/example_face_sampleDetectLandmarksvideo: modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/build.make
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_face.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_photo.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_objdetect.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_highgui.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_calib3d.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_features2d.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_flann.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_videoio.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_imgcodecs.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_imgproc.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: lib/libopencv_core.4.5.0.dylib
bin/example_face_sampleDetectLandmarksvideo: modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/example_face_sampleDetectLandmarksvideo"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/face && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/build: bin/example_face_sampleDetectLandmarksvideo

.PHONY : modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/build

modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/face && $(CMAKE_COMMAND) -P CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/cmake_clean.cmake
.PHONY : modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/clean

modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/face /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/face /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/face/CMakeFiles/example_face_sampleDetectLandmarksvideo.dir/depend

