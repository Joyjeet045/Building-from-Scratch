#[[
inference_engine CMake module. Each subdirectory under src/ contributes its own
source list via engine_add_sources, so adding a file touches only that module's
CMakeLists.txt. Release flags mirror the ones used in the matrix-multiplication
write-up so the tuned GEMM kernels get vectorized.
]]
add_library(engine_core STATIC)

#[[ Appends a module's sources to engine_core, paths relative to that module. ]]
function(engine_add_sources)
    set(resolved "")
    foreach(source ${ARGN})
        list(APPEND resolved "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    endforeach()
    target_sources(engine_core PRIVATE ${resolved})
endfunction()

target_include_directories(engine_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src/core)
add_subdirectory(src/graph)
add_subdirectory(src/model)
add_subdirectory(src/kernels)
add_subdirectory(src/providers)
add_subdirectory(src/runtime)
add_subdirectory(src/serving)

find_package(Threads REQUIRED)
target_link_libraries(engine_core PUBLIC Threads::Threads)

if(WIN32)
    target_link_libraries(engine_core PUBLIC ws2_32)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(engine_core PRIVATE -Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
            target_compile_options(engine_core PRIVATE -march=native)
        endif()
        target_compile_options(engine_core PRIVATE -funroll-loops)
    endif()
endif()

add_subdirectory(src/cli)
