cmake_minimum_required(VERSION 3.10)

# set the project name
project(AXPY)

find_package(Kokkos REQUIRED)

# add the executable
add_executable(axpy main.cpp)
target_link_libraries(axpy Kokkos::kokkos)