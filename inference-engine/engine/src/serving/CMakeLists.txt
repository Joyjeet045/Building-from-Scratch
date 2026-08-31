#[[ serving: request batching and the HTTP front door ]]
engine_add_sources(
    dynamic_batcher.cpp
    http_server.cpp
)
