# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.13.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.13.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Volumes/MacintoshHD/Users/aclapes/Code/HOG

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Volumes/MacintoshHD/Users/aclapes/Code/HOG/test_performance

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Volumes/MacintoshHD/Users/aclapes/Code/HOG/test_performance/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main.cpp.o -c /Volumes/MacintoshHD/Users/aclapes/Code/HOG/main.cpp

CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Volumes/MacintoshHD/Users/aclapes/Code/HOG/main.cpp > CMakeFiles/main.dir/main.cpp.i

CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Volumes/MacintoshHD/Users/aclapes/Code/HOG/main.cpp -o CMakeFiles/main.dir/main.cpp.s

CMakeFiles/main.dir/HOG.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/HOG.cpp.o: ../HOG.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Volumes/MacintoshHD/Users/aclapes/Code/HOG/test_performance/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/HOG.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/HOG.cpp.o -c /Volumes/MacintoshHD/Users/aclapes/Code/HOG/HOG.cpp

CMakeFiles/main.dir/HOG.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/HOG.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Volumes/MacintoshHD/Users/aclapes/Code/HOG/HOG.cpp > CMakeFiles/main.dir/HOG.cpp.i

CMakeFiles/main.dir/HOG.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/HOG.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Volumes/MacintoshHD/Users/aclapes/Code/HOG/HOG.cpp -o CMakeFiles/main.dir/HOG.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o" \
"CMakeFiles/main.dir/HOG.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/main.cpp.o
main: CMakeFiles/main.dir/HOG.cpp.o
main: CMakeFiles/main.dir/build.make
main: /usr/local/lib/libopencv_stitching.3.4.3.dylib
main: /usr/local/lib/libopencv_superres.3.4.3.dylib
main: /usr/local/lib/libopencv_videostab.3.4.3.dylib
main: /usr/local/lib/libopencv_aruco.3.4.3.dylib
main: /usr/local/lib/libopencv_bgsegm.3.4.3.dylib
main: /usr/local/lib/libopencv_bioinspired.3.4.3.dylib
main: /usr/local/lib/libopencv_ccalib.3.4.3.dylib
main: /usr/local/lib/libopencv_dnn_objdetect.3.4.3.dylib
main: /usr/local/lib/libopencv_dpm.3.4.3.dylib
main: /usr/local/lib/libopencv_face.3.4.3.dylib
main: /usr/local/lib/libopencv_fuzzy.3.4.3.dylib
main: /usr/local/lib/libopencv_hfs.3.4.3.dylib
main: /usr/local/lib/libopencv_img_hash.3.4.3.dylib
main: /usr/local/lib/libopencv_line_descriptor.3.4.3.dylib
main: /usr/local/lib/libopencv_optflow.3.4.3.dylib
main: /usr/local/lib/libopencv_reg.3.4.3.dylib
main: /usr/local/lib/libopencv_rgbd.3.4.3.dylib
main: /usr/local/lib/libopencv_saliency.3.4.3.dylib
main: /usr/local/lib/libopencv_stereo.3.4.3.dylib
main: /usr/local/lib/libopencv_structured_light.3.4.3.dylib
main: /usr/local/lib/libopencv_surface_matching.3.4.3.dylib
main: /usr/local/lib/libopencv_tracking.3.4.3.dylib
main: /usr/local/lib/libopencv_xfeatures2d.3.4.3.dylib
main: /usr/local/lib/libopencv_ximgproc.3.4.3.dylib
main: /usr/local/lib/libopencv_xobjdetect.3.4.3.dylib
main: /usr/local/lib/libopencv_xphoto.3.4.3.dylib
main: /usr/local/lib/libopencv_shape.3.4.3.dylib
main: /usr/local/lib/libopencv_photo.3.4.3.dylib
main: /usr/local/lib/libopencv_calib3d.3.4.3.dylib
main: /usr/local/lib/libopencv_features2d.3.4.3.dylib
main: /usr/local/lib/libopencv_highgui.3.4.3.dylib
main: /usr/local/lib/libopencv_videoio.3.4.3.dylib
main: /usr/local/lib/libopencv_phase_unwrapping.3.4.3.dylib
main: /usr/local/lib/libopencv_dnn.3.4.3.dylib
main: /usr/local/lib/libopencv_video.3.4.3.dylib
main: /usr/local/lib/libopencv_datasets.3.4.3.dylib
main: /usr/local/lib/libopencv_flann.3.4.3.dylib
main: /usr/local/lib/libopencv_ml.3.4.3.dylib
main: /usr/local/lib/libopencv_plot.3.4.3.dylib
main: /usr/local/lib/libopencv_imgcodecs.3.4.3.dylib
main: /usr/local/lib/libopencv_objdetect.3.4.3.dylib
main: /usr/local/lib/libopencv_imgproc.3.4.3.dylib
main: /usr/local/lib/libopencv_core.3.4.3.dylib
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Volumes/MacintoshHD/Users/aclapes/Code/HOG/test_performance/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /Volumes/MacintoshHD/Users/aclapes/Code/HOG/test_performance && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Volumes/MacintoshHD/Users/aclapes/Code/HOG /Volumes/MacintoshHD/Users/aclapes/Code/HOG /Volumes/MacintoshHD/Users/aclapes/Code/HOG/test_performance /Volumes/MacintoshHD/Users/aclapes/Code/HOG/test_performance /Volumes/MacintoshHD/Users/aclapes/Code/HOG/test_performance/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

