#[[
Root CMake build script: wires together the engine library/executable and
the test suite.
]]
cmake_minimum_required(VERSION 3.15)
project(InferenceEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Keep binaries at a stable path regardless of which module subdirectory
# declares them.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(inference_engine)

enable_testing()
add_subdirectory(tests)
