cmake_minimum_required(VERSION 3.10)

# set the project name
project(DOT)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(TBB REQUIRED)

# add the executable
add_executable(dot main.cpp)
target_link_libraries(dot PRIVATE TBB::tbb)