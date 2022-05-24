cmake_minimum_required(VERSION 3.21.0)

# set the project name
project(AXPY LANGUAGES CXX)

# find OneAPI
find_package(IntelDPCPP REQUIRED)
find_package(MKL REQUIRED)

# add the executable
add_executable(axpy main.cpp)
target_link_libraries(axpy MKL::MKL)