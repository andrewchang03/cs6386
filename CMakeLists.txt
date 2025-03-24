cmake_minimum_required(VERSION 3.20)
project(caboose LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
  message(FATAL_ERROR "Could not find CUDA.")
endif()

# Set CUDA toolkit path explicitly if necessary (optional)
set(CUSPARSE_LIB /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64)

# Executable
file(GLOB SOURCES "main.cu")
add_executable(main ${SOURCES})

# Compiler and linker flags (if needed)
target_compile_options(main PRIVATE -fsanitize=address,leak)
target_link_options(main PRIVATE -fsanitize=address,leak)

# Link CUDA libraries
target_link_libraries(main PRIVATE cublas cusparse)

# If CUDA toolkit paths are needed, explicitly add include and library directories
target_include_directories(main PRIVATE)
target_link_directories(main PRIVATE /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 cusparse)