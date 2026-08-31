#[[ cli: the engine_exe front end ]]
add_executable(engine_exe main.cpp)
target_link_libraries(engine_exe PRIVATE engine_core)
