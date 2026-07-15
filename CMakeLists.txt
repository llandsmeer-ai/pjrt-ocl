# Root CMake for the pip/scikit-build-core build: build libpjrt_ocl.so and
# install it INTO the wheel next to the python package (pjrt_ocl/), where the
# package's find_plugin_library() looks for it. Dev builds still use
# `cmake -S pjrt_plugin -B pjrt_plugin/build` directly.
cmake_minimum_required(VERSION 3.22)
project(pjrt_ocl_wheel CXX)

set(PJRT_OCL_BUILD_TESTS OFF CACHE BOOL "" FORCE)  # no runtime_test in the wheel
add_subdirectory(pjrt_plugin)

# Land the shared library at pjrt_ocl/libpjrt_ocl.so inside the wheel.
install(TARGETS pjrt_ocl LIBRARY DESTINATION pjrt_ocl
                         RUNTIME DESTINATION pjrt_ocl)
