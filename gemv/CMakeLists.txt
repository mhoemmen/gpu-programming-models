cmake_minimum_required(VERSION 3.10)

# set the project name
project(GEMV)

find_package(Kokkos REQUIRED)

# add the executable
add_executable(gemv main.cpp)
target_link_libraries(gemv Kokkos::kokkos)