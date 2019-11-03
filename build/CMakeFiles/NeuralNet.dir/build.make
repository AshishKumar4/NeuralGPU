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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/shruti/Desktop/PDC/FNN Test Beds CUDA"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build"

# Include any dependencies generated for this target.
include CMakeFiles/NeuralNet.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/NeuralNet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NeuralNet.dir/flags.make

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.o: ../FNN\ Test\ Beds\ CUDA/kernel.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/FNN Test Beds CUDA/kernel.cu" -o CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.o

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.o: ../FNN\ Test\ Beds\ CUDA/dot_adder.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/FNN Test Beds CUDA/dot_adder.cu" -o CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.o

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.o: ../FNN\ Test\ Beds\ CUDA/Neuron.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/FNN Test Beds CUDA/Neuron.cu" -o CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.o

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.o: ../FNN\ Test\ Beds\ CUDA/threads.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/FNN Test Beds CUDA/threads.cu" -o CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.o

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.o: ../FNN\ Test\ Beds\ CUDA/idx.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/FNN Test Beds CUDA/idx.cu" -o CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.o

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.o: ../FNN\ Test\ Beds\ CUDA/usableLib.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.o -c "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/FNN Test Beds CUDA/usableLib.cpp"

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/FNN Test Beds CUDA/usableLib.cpp" > CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.i

CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/FNN Test Beds CUDA/usableLib.cpp" -o CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.s

# Object files for target NeuralNet
NeuralNet_OBJECTS = \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.o"

# External object files for target NeuralNet
NeuralNet_EXTERNAL_OBJECTS =

CMakeFiles/NeuralNet.dir/cmake_device_link.o: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.o
CMakeFiles/NeuralNet.dir/cmake_device_link.o: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.o
CMakeFiles/NeuralNet.dir/cmake_device_link.o: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.o
CMakeFiles/NeuralNet.dir/cmake_device_link.o: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.o
CMakeFiles/NeuralNet.dir/cmake_device_link.o: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.o
CMakeFiles/NeuralNet.dir/cmake_device_link.o: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.o
CMakeFiles/NeuralNet.dir/cmake_device_link.o: CMakeFiles/NeuralNet.dir/build.make
CMakeFiles/NeuralNet.dir/cmake_device_link.o: CMakeFiles/NeuralNet.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Linking CUDA device code CMakeFiles/NeuralNet.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NeuralNet.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NeuralNet.dir/build: CMakeFiles/NeuralNet.dir/cmake_device_link.o

.PHONY : CMakeFiles/NeuralNet.dir/build

# Object files for target NeuralNet
NeuralNet_OBJECTS = \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.o" \
"CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.o"

# External object files for target NeuralNet
NeuralNet_EXTERNAL_OBJECTS =

NeuralNet: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/kernel.cu.o
NeuralNet: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/dot_adder.cu.o
NeuralNet: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/Neuron.cu.o
NeuralNet: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/threads.cu.o
NeuralNet: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/idx.cu.o
NeuralNet: CMakeFiles/NeuralNet.dir/FNN_Test_Beds_CUDA/usableLib.cpp.o
NeuralNet: CMakeFiles/NeuralNet.dir/build.make
NeuralNet: CMakeFiles/NeuralNet.dir/cmake_device_link.o
NeuralNet: CMakeFiles/NeuralNet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable NeuralNet"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NeuralNet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NeuralNet.dir/build: NeuralNet

.PHONY : CMakeFiles/NeuralNet.dir/build

CMakeFiles/NeuralNet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NeuralNet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NeuralNet.dir/clean

CMakeFiles/NeuralNet.dir/depend:
	cd "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/shruti/Desktop/PDC/FNN Test Beds CUDA" "/home/shruti/Desktop/PDC/FNN Test Beds CUDA" "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build" "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build" "/home/shruti/Desktop/PDC/FNN Test Beds CUDA/build/CMakeFiles/NeuralNet.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/NeuralNet.dir/depend

