cmake_minimum_required(VERSION 3.10)

enable_language(Fortran)

# set the project name
project(AXPY)

# add the executable
add_executable(axpy main.f90)