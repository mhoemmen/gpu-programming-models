cmake_minimum_required(VERSION 3.10)

# set the project name
project(DOT)

find_package(Kokkos REQUIRED)

# add the executable
add_executable(dot main.cpp)
target_link_libraries(dot Kokkos::kokkos)