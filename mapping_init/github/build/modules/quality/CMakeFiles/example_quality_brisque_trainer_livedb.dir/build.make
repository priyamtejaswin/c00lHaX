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
include modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/depend.make

# Include the progress variables for this target.
include modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/progress.make

# Include the compile flags for this target's objects.
include modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/flags.make

modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.o: modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/flags.make
modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality/samples/brisque_trainer_livedb.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality/samples/brisque_trainer_livedb.cpp

modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality/samples/brisque_trainer_livedb.cpp > CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.i

modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality/samples/brisque_trainer_livedb.cpp -o CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.s

# Object files for target example_quality_brisque_trainer_livedb
example_quality_brisque_trainer_livedb_OBJECTS = \
"CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.o"

# External object files for target example_quality_brisque_trainer_livedb
example_quality_brisque_trainer_livedb_EXTERNAL_OBJECTS =

bin/example_quality_brisque_trainer_livedb: modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/samples/brisque_trainer_livedb.cpp.o
bin/example_quality_brisque_trainer_livedb: modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/build.make
bin/example_quality_brisque_trainer_livedb: lib/libopencv_quality.4.5.0.dylib
bin/example_quality_brisque_trainer_livedb: lib/libopencv_ml.4.5.0.dylib
bin/example_quality_brisque_trainer_livedb: lib/libopencv_highgui.4.5.0.dylib
bin/example_quality_brisque_trainer_livedb: lib/libopencv_videoio.4.5.0.dylib
bin/example_quality_brisque_trainer_livedb: lib/libopencv_imgcodecs.4.5.0.dylib
bin/example_quality_brisque_trainer_livedb: lib/libopencv_imgproc.4.5.0.dylib
bin/example_quality_brisque_trainer_livedb: lib/libopencv_core.4.5.0.dylib
bin/example_quality_brisque_trainer_livedb: modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/example_quality_brisque_trainer_livedb"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_quality_brisque_trainer_livedb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/build: bin/example_quality_brisque_trainer_livedb

.PHONY : modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/build

modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality && $(CMAKE_COMMAND) -P CMakeFiles/example_quality_brisque_trainer_livedb.dir/cmake_clean.cmake
.PHONY : modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/clean

modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/quality /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/quality/CMakeFiles/example_quality_brisque_trainer_livedb.dir/depend

