cmake_minimum_required(VERSION 3.10)

# set the project name
project(RAY)

# add the executable
add_executable(ray main.cpp)
add_executable(no-virtual-ray no-virtual.cpp)
