#[[
tests CMake module: builds engine_tests, one executable containing the custom
test-framework runner plus every module's test translation units. Test folders
mirror the engine's module layout.
]]
add_executable(engine_tests
    framework/test_main.cpp

    core/test_tensor.cpp
    core/test_allocator.cpp
    core/test_thread_pool.cpp

    graph/test_graph.cpp
    graph/test_execution_levels.cpp
    graph/test_optimizer.cpp

    model/test_model_loader.cpp

    kernels/test_gemm.cpp
    kernels/test_operators.cpp

    runtime/test_config.cpp
    runtime/test_batching.cpp
    runtime/test_parallel_execution.cpp
    runtime/test_inference_engine.cpp

    serving/test_dynamic_batcher.cpp
)
target_link_libraries(engine_tests PRIVATE engine_core)
target_include_directories(engine_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME engine_tests COMMAND engine_tests)
