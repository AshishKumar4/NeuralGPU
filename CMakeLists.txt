cmake_minimum_required(VERSION 3.0.0)
project(NeuralNet LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 17)

# Include the Necessary Headers
include_directories(./
    ./FNN\ Test\ Beds\ CUDA/
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Build our Executable
set(BASE_SOURCES ./FNN\ Test\ Beds\ CUDA/kernel.cu
./FNN\ Test\ Beds\ CUDA/dot_adder.cu
./FNN\ Test\ Beds\ CUDA/Neuron.cu
./FNN\ Test\ Beds\ CUDA/threads.cu
./FNN\ Test\ Beds\ CUDA/idx.cu
./FNN\ Test\ Beds\ CUDA/usableLib.cpp
)

set(CMAKE_CXX_FLAGS "-O2 -fpic -w")
set(CMAKE_C_FLAGS "-O2 -fpic -w")

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=clang++-8")

add_executable(${PROJECT_NAME} ${BASE_SOURCES})