cmake_minimum_required(VERSION 3.10)

# set the project name
project(GEMV)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(TBB REQUIRED)

# add the executable
add_executable(gemv main.cpp)
target_link_libraries(gemv PRIVATE TBB::tbb)