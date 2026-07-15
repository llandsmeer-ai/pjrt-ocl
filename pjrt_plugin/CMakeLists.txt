cmake_minimum_required(VERSION 3.22)
project(pjrt_ocl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCL REQUIRED)

# Embed kernels/vm.cl as a C string (recompiled when the .cl changes).
set(VM_CL_IN ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vm2.cl)
set(VM_CL_OUT ${CMAKE_CURRENT_BINARY_DIR}/vm_cl_source.h)
add_custom_command(
  OUTPUT ${VM_CL_OUT}
  COMMAND ${CMAKE_COMMAND} -DIN=${VM_CL_IN} -DOUT=${VM_CL_OUT}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cl.cmake
  DEPENDS ${VM_CL_IN} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cl.cmake
  COMMENT "Embedding vm.cl")

add_library(pjrt_ocl SHARED src/plugin.cc src/runtime.cc ${VM_CL_OUT})
target_include_directories(pjrt_ocl PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(pjrt_ocl PRIVATE OpenCL::OpenCL)
target_compile_options(pjrt_ocl PRIVATE -Wall -fvisibility=hidden)
set_target_properties(pjrt_ocl PROPERTIES PREFIX "lib" OUTPUT_NAME "pjrt_ocl")

# Standalone runtime test: hand-encoded VMProgram, no PJRT/python involved.
add_executable(runtime_test src/runtime_test.cc src/runtime.cc ${VM_CL_OUT})
target_include_directories(runtime_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(runtime_test PRIVATE OpenCL::OpenCL)
