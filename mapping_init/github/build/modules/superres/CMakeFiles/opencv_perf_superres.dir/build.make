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
include modules/superres/CMakeFiles/opencv_perf_superres.dir/depend.make

# Include the progress variables for this target.
include modules/superres/CMakeFiles/opencv_perf_superres.dir/progress.make

# Include the compile flags for this target's objects.
include modules/superres/CMakeFiles/opencv_perf_superres.dir/flags.make

modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.o: modules/superres/CMakeFiles/opencv_perf_superres.dir/flags.make
modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres/perf/perf_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres/perf/perf_main.cpp

modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres/perf/perf_main.cpp > CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.i

modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres/perf/perf_main.cpp -o CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.s

modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.o: modules/superres/CMakeFiles/opencv_perf_superres.dir/flags.make
modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres/perf/perf_superres.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres/perf/perf_superres.cpp

modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres/perf/perf_superres.cpp > CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.i

modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres/perf/perf_superres.cpp -o CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.s

# Object files for target opencv_perf_superres
opencv_perf_superres_OBJECTS = \
"CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.o" \
"CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.o"

# External object files for target opencv_perf_superres
opencv_perf_superres_EXTERNAL_OBJECTS =

bin/opencv_perf_superres: modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.o
bin/opencv_perf_superres: modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.o
bin/opencv_perf_superres: modules/superres/CMakeFiles/opencv_perf_superres.dir/build.make
bin/opencv_perf_superres: lib/libopencv_ts.a
bin/opencv_perf_superres: lib/libopencv_superres.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_highgui.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_videoio.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_optflow.4.5.0.dylib
bin/opencv_perf_superres: 3rdparty/lib/libippiw.a
bin/opencv_perf_superres: 3rdparty/ippicv/ippicv_mac/icv/lib/intel64/libippicv.a
bin/opencv_perf_superres: lib/libopencv_ximgproc.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_imgcodecs.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_video.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_calib3d.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_features2d.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_flann.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_imgproc.4.5.0.dylib
bin/opencv_perf_superres: lib/libopencv_core.4.5.0.dylib
bin/opencv_perf_superres: modules/superres/CMakeFiles/opencv_perf_superres.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../bin/opencv_perf_superres"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_perf_superres.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/superres/CMakeFiles/opencv_perf_superres.dir/build: bin/opencv_perf_superres

.PHONY : modules/superres/CMakeFiles/opencv_perf_superres.dir/build

modules/superres/CMakeFiles/opencv_perf_superres.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres && $(CMAKE_COMMAND) -P CMakeFiles/opencv_perf_superres.dir/cmake_clean.cmake
.PHONY : modules/superres/CMakeFiles/opencv_perf_superres.dir/clean

modules/superres/CMakeFiles/opencv_perf_superres.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/superres /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/superres/CMakeFiles/opencv_perf_superres.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/superres/CMakeFiles/opencv_perf_superres.dir/depend

