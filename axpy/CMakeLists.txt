cmake_minimum_required(VERSION 3.10)

# set the project name
project(AXPY LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)
find_package(chai REQUIRED)
find_package(umpire REQUIRED)
find_package(RAJA REQUIRED)

include_directories(${UMPIRE_INCLUDE_DIRS} ${CHAI_INCLUDE_DIRS})

# Work around CHAI issue #214
get_target_property(chai_interface_libs chai INTERFACE_LINK_LIBRARIES)
list(REMOVE_ITEM chai_interface_libs cuda)
set_target_properties(chai PROPERTIES INTERFACE_LINK_LIBRARIES ${chai_interface_libs})

# add the executable
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(axpy main.cpp)
target_link_libraries(axpy RAJA chai)