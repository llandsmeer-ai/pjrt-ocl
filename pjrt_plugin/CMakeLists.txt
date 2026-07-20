cmake_minimum_required(VERSION 3.22)
project(pjrt_ocl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCL REQUIRED)

# Embed the modular OpenCL kernel (per-op-family files, concatenated in order)
# as a C string. Add a new op family by dropping kernels/ops/<name>.cl and
# listing it here (before vm_main.cl).
set(VM_CL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vm_common.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/ew.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/gather.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/reduce.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/mma.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/iota.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/scatter.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/dynslice.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/reduce_window.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/region.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vm_main.cl)
set(VM_CL_OUT ${CMAKE_CURRENT_BINARY_DIR}/vm_cl_source.h)
# Join with | so the list survives as ONE argument to the -P script (a bare
# ;-list would expand into separate cmake args).
string(REPLACE ";" "," VM_CL_SOURCES_STR "${VM_CL_SOURCES}")
add_custom_command(
  OUTPUT ${VM_CL_OUT}
  COMMAND ${CMAKE_COMMAND} -DSRCS=${VM_CL_SOURCES_STR} -DOUT=${VM_CL_OUT}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cl.cmake
  DEPENDS ${VM_CL_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cl.cmake
  COMMENT "Embedding modular VM kernel")

add_library(pjrt_ocl SHARED src/plugin.cc src/runtime.cc ${VM_CL_OUT})
target_include_directories(pjrt_ocl PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(pjrt_ocl PRIVATE OpenCL::OpenCL)
target_compile_options(pjrt_ocl PRIVATE -Wall -fvisibility=hidden)
set_target_properties(pjrt_ocl PROPERTIES PREFIX "lib" OUTPUT_NAME "pjrt_ocl")

# Standalone runtime test: hand-encoded VMProgram, no PJRT/python involved.
# Skipped for the pip/wheel build (root CMakeLists sets this OFF).
option(PJRT_OCL_BUILD_TESTS "Build the C++ runtime_test executable" ON)
if(PJRT_OCL_BUILD_TESTS)
  add_executable(runtime_test src/runtime_test.cc src/runtime.cc ${VM_CL_OUT})
  target_include_directories(runtime_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(runtime_test PRIVATE OpenCL::OpenCL)
endif()
