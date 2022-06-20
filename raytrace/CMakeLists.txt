cmake_minimum_required(VERSION 3.10)

# set the project name
project(RAY LANGUAGES CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# add the executable
add_executable(ray main.cu)
