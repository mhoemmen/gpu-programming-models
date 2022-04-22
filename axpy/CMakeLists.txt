cmake_minimum_required(VERSION 3.10)

# set the project name
project(AXPY)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(TBB REQUIRED)

# add the executable
add_executable(axpy main.cpp)
target_link_libraries(axpy PRIVATE TBB::tbb)