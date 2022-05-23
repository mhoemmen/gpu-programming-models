cmake_minimum_required(VERSION 3.10)

# set the project name
project(AXPY LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

# add the executable
add_executable(axpy main.cu)
target_link_libraries(axpy CUDA::cublas)