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
include modules/text/CMakeFiles/example_text_character_recognition.dir/depend.make

# Include the progress variables for this target.
include modules/text/CMakeFiles/example_text_character_recognition.dir/progress.make

# Include the compile flags for this target's objects.
include modules/text/CMakeFiles/example_text_character_recognition.dir/flags.make

modules/text/CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.o: modules/text/CMakeFiles/example_text_character_recognition.dir/flags.make
modules/text/CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.o: /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/text/samples/character_recognition.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/text/CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.o"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/text && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.o -c /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/text/samples/character_recognition.cpp

modules/text/CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.i"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/text && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/text/samples/character_recognition.cpp > CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.i

modules/text/CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.s"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/text && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/text/samples/character_recognition.cpp -o CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.s

# Object files for target example_text_character_recognition
example_text_character_recognition_OBJECTS = \
"CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.o"

# External object files for target example_text_character_recognition
example_text_character_recognition_EXTERNAL_OBJECTS =

bin/example_text_character_recognition: modules/text/CMakeFiles/example_text_character_recognition.dir/samples/character_recognition.cpp.o
bin/example_text_character_recognition: modules/text/CMakeFiles/example_text_character_recognition.dir/build.make
bin/example_text_character_recognition: lib/libopencv_text.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_ml.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_dnn.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_features2d.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_highgui.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_flann.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_videoio.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_imgcodecs.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_imgproc.4.5.0.dylib
bin/example_text_character_recognition: lib/libopencv_core.4.5.0.dylib
bin/example_text_character_recognition: modules/text/CMakeFiles/example_text_character_recognition.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/example_text_character_recognition"
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/text && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_text_character_recognition.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/text/CMakeFiles/example_text_character_recognition.dir/build: bin/example_text_character_recognition

.PHONY : modules/text/CMakeFiles/example_text_character_recognition.dir/build

modules/text/CMakeFiles/example_text_character_recognition.dir/clean:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/text && $(CMAKE_COMMAND) -P CMakeFiles/example_text_character_recognition.dir/cmake_clean.cmake
.PHONY : modules/text/CMakeFiles/example_text_character_recognition.dir/clean

modules/text/CMakeFiles/example_text_character_recognition.dir/depend:
	cd /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv-master /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/opencv_contrib-master/modules/text /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/text /Users/tejaswin.p/THIS_LAPTOP_projects/c00lHaX/mapping_init/github/build/modules/text/CMakeFiles/example_text_character_recognition.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/text/CMakeFiles/example_text_character_recognition.dir/depend

