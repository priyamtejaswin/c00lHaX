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

# Utility rule file for opencv_samples_alphamat.

# Include the progress variables for this target.
include modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/progress.make

opencv_samples_alphamat: modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/build.make

.PHONY : opencv_samples_alphamat

# Rule to build all files generated by this target.
modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/build: opencv_samples_alphamat

.PHONY : modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/build

modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/alphamat && $(CMAKE_COMMAND) -P CMakeFiles/opencv_samples_alphamat.dir/cmake_clean.cmake
.PHONY : modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/clean

modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/alphamat /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/alphamat /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/alphamat/CMakeFiles/opencv_samples_alphamat.dir/depend

