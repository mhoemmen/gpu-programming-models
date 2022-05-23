cmake_minimum_required(VERSION 3.10)

# set the project name
project(AXPY LANGUAGES CUDA)

# add the executables
add_executable(axpy main.cu)
add_executable(thrust_axpy thrust.cu)