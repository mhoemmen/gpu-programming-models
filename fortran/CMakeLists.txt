cmake_minimum_required(VERSION 3.10)

enable_language(Fortran CXX)

find_package(flcl REQUIRED)
find_package(Kokkos REQUIRED)

# set the project name
project(AXPY)

# add the executable
add_executable(axpy main.f90 axpy-view-f.f90 axpy-view-cxx.cc)
target_link_libraries(axpy flcl::flcl)