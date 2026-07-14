# Standalone PoC build — NOT part of a (future) top-level build system.
# cmake -S . -B build -G Ninja && cmake --build build
cmake_minimum_required(VERSION 3.20)
project(pjrt_ocl_skeleton CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(OpenCL REQUIRED)

add_library(pjrt_ocl_skeleton SHARED plugin.cc)
target_link_libraries(pjrt_ocl_skeleton PRIVATE OpenCL::OpenCL)
target_compile_options(pjrt_ocl_skeleton PRIVATE -Wall -Wextra -fvisibility=hidden)
# Only GetPjRtApi is exported (visibility("default") in source).
