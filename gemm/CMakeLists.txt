cmake_minimum_required(VERSION 3.10)

# set the project name
project(GEMM)

find_package(BLAS REQUIRED)

# add the executable
add_executable(gemm main.cpp)
target_link_libraries(gemm ${BLAS_LIBRARIES})