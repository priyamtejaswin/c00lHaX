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
include modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/depend.make

# Include the progress variables for this target.
include modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/progress.make

# Include the compile flags for this target's objects.
include modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/flags.make

modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.o: modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/flags.make
modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform/test/test_intensity_transform.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform/test/test_intensity_transform.cpp

modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform/test/test_intensity_transform.cpp > CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.i

modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform/test/test_intensity_transform.cpp -o CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.s

modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.o: modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/flags.make
modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform/test/test_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform/test/test_main.cpp

modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform/test/test_main.cpp > CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.i

modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform/test/test_main.cpp -o CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.s

# Object files for target opencv_test_intensity_transform
opencv_test_intensity_transform_OBJECTS = \
"CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.o" \
"CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.o"

# External object files for target opencv_test_intensity_transform
opencv_test_intensity_transform_EXTERNAL_OBJECTS =

bin/opencv_test_intensity_transform: modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_intensity_transform.cpp.o
bin/opencv_test_intensity_transform: modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/test/test_main.cpp.o
bin/opencv_test_intensity_transform: modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/build.make
bin/opencv_test_intensity_transform: lib/libopencv_ts.a
bin/opencv_test_intensity_transform: lib/libopencv_intensity_transform.4.5.0.dylib
bin/opencv_test_intensity_transform: lib/libopencv_highgui.4.5.0.dylib
bin/opencv_test_intensity_transform: 3rdparty/lib/libippiw.a
bin/opencv_test_intensity_transform: 3rdparty/ippicv/ippicv_mac/icv/lib/intel64/libippicv.a
bin/opencv_test_intensity_transform: lib/libopencv_videoio.4.5.0.dylib
bin/opencv_test_intensity_transform: lib/libopencv_imgcodecs.4.5.0.dylib
bin/opencv_test_intensity_transform: lib/libopencv_imgproc.4.5.0.dylib
bin/opencv_test_intensity_transform: lib/libopencv_core.4.5.0.dylib
bin/opencv_test_intensity_transform: modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../bin/opencv_test_intensity_transform"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_test_intensity_transform.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/build: bin/opencv_test_intensity_transform

.PHONY : modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/build

modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform && $(CMAKE_COMMAND) -P CMakeFiles/opencv_test_intensity_transform.dir/cmake_clean.cmake
.PHONY : modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/clean

modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/intensity_transform /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/intensity_transform/CMakeFiles/opencv_test_intensity_transform.dir/depend

