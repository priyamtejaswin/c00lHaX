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
include modules/gapi/CMakeFiles/example_gapi_draw_example.dir/depend.make

# Include the progress variables for this target.
include modules/gapi/CMakeFiles/example_gapi_draw_example.dir/progress.make

# Include the compile flags for this target's objects.
include modules/gapi/CMakeFiles/example_gapi_draw_example.dir/flags.make

modules/gapi/CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.o: modules/gapi/CMakeFiles/example_gapi_draw_example.dir/flags.make
modules/gapi/CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/modules/gapi/samples/draw_example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/gapi/CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/gapi && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/modules/gapi/samples/draw_example.cpp

modules/gapi/CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/gapi && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/modules/gapi/samples/draw_example.cpp > CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.i

modules/gapi/CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/gapi && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/modules/gapi/samples/draw_example.cpp -o CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.s

# Object files for target example_gapi_draw_example
example_gapi_draw_example_OBJECTS = \
"CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.o"

# External object files for target example_gapi_draw_example
example_gapi_draw_example_EXTERNAL_OBJECTS =

bin/example_gapi_draw_example: modules/gapi/CMakeFiles/example_gapi_draw_example.dir/samples/draw_example.cpp.o
bin/example_gapi_draw_example: modules/gapi/CMakeFiles/example_gapi_draw_example.dir/build.make
bin/example_gapi_draw_example: lib/libopencv_gapi.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_video.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_highgui.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_calib3d.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_features2d.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_flann.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_videoio.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_imgcodecs.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_imgproc.4.5.0.dylib
bin/example_gapi_draw_example: lib/libopencv_core.4.5.0.dylib
bin/example_gapi_draw_example: modules/gapi/CMakeFiles/example_gapi_draw_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/example_gapi_draw_example"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/gapi && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_gapi_draw_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/gapi/CMakeFiles/example_gapi_draw_example.dir/build: bin/example_gapi_draw_example

.PHONY : modules/gapi/CMakeFiles/example_gapi_draw_example.dir/build

modules/gapi/CMakeFiles/example_gapi_draw_example.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/gapi && $(CMAKE_COMMAND) -P CMakeFiles/example_gapi_draw_example.dir/cmake_clean.cmake
.PHONY : modules/gapi/CMakeFiles/example_gapi_draw_example.dir/clean

modules/gapi/CMakeFiles/example_gapi_draw_example.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master/modules/gapi /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/gapi /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/gapi/CMakeFiles/example_gapi_draw_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/gapi/CMakeFiles/example_gapi_draw_example.dir/depend

