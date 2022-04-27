cmake_minimum_required(VERSION 3.10)

# set the project name
project(AXPY)

find_package(hip)

# add the executable
add_executable(axpy main.cpp)
target_link_libraries(axpy hip::device)
