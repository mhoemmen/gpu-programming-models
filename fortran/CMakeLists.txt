cmake_minimum_required(VERSION 3.10)

enable_language(Fortran)

find_package(CUDAToolkit REQUIRED)

# set the project name
project(AXPY)

# add the executable
set_source_files_properties(main.cuf PROPERTIES LANGUAGE Fortran CMAKE_Fortran_FORMAT FREE)
add_executable(axpy main.cuf)
target_link_libraries(axpy CUDA::toolkit)