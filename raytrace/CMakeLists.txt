cmake_minimum_required(VERSION 3.10)

# set the project name
project(RAY)

find_package(Kokkos REQUIRED)

# add the executable
add_executable(ray main.cpp)
target_link_libraries(ray Kokkos::kokkos)