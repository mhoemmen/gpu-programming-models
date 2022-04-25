cmake_minimum_required(VERSION 3.10)

# set the project name
project(AXPY LANGUAGES CUDA)

# add the executable
add_executable(axpy main.cu)