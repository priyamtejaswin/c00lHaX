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
include modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/depend.make

# Include the progress variables for this target.
include modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/progress.make

# Include the compile flags for this target's objects.
include modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/flags.make

modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.o: modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/flags.make
modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality/samples/brisque_eval_tid2008.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality/samples/brisque_eval_tid2008.cpp

modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality/samples/brisque_eval_tid2008.cpp > CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.i

modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality/samples/brisque_eval_tid2008.cpp -o CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.s

# Object files for target example_quality_brisque_eval_tid2008
example_quality_brisque_eval_tid2008_OBJECTS = \
"CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.o"

# External object files for target example_quality_brisque_eval_tid2008
example_quality_brisque_eval_tid2008_EXTERNAL_OBJECTS =

bin/example_quality_brisque_eval_tid2008: modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/samples/brisque_eval_tid2008.cpp.o
bin/example_quality_brisque_eval_tid2008: modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/build.make
bin/example_quality_brisque_eval_tid2008: lib/libopencv_quality.4.5.0.dylib
bin/example_quality_brisque_eval_tid2008: lib/libopencv_ml.4.5.0.dylib
bin/example_quality_brisque_eval_tid2008: lib/libopencv_highgui.4.5.0.dylib
bin/example_quality_brisque_eval_tid2008: lib/libopencv_videoio.4.5.0.dylib
bin/example_quality_brisque_eval_tid2008: lib/libopencv_imgcodecs.4.5.0.dylib
bin/example_quality_brisque_eval_tid2008: lib/libopencv_imgproc.4.5.0.dylib
bin/example_quality_brisque_eval_tid2008: lib/libopencv_core.4.5.0.dylib
bin/example_quality_brisque_eval_tid2008: modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/example_quality_brisque_eval_tid2008"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_quality_brisque_eval_tid2008.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/build: bin/example_quality_brisque_eval_tid2008

.PHONY : modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/build

modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && $(CMAKE_COMMAND) -P CMakeFiles/example_quality_brisque_eval_tid2008.dir/cmake_clean.cmake
.PHONY : modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/clean

modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/quality/CMakeFiles/example_quality_brisque_eval_tid2008.dir/depend

