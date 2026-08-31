/*
 * http_server.h
 * Minimal HTTP/1.1 inference server built directly on sockets so the project
 * keeps its zero-dependency property. Requests are handed to the dynamic
 * batcher rather than executed inline.
 *
 * Routes:
 *   GET  /health   liveness probe
 *   GET  /metrics  request counters and batching statistics
 *   POST /infer    body is exactly featureCount little-endian float32 values,
 *                  or featureCount raw uint8 pixels; responds with JSON
 *                  {"prediction": n, "logits": [...]}
 *
 * All request sizes are bounded before any allocation and unknown routes are
 * rejected, so a malformed or hostile request cannot drive memory growth.
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <string>

#include "ie/serving/dynamic_batcher.h"

namespace ie {

struct ServerStats {
    std::atomic<uint64_t> requests{0};
    std::atomic<uint64_t> errors{0};
};

class HttpServer {
public:
    HttpServer(DynamicBatcher& batcher, std::string address, int port, std::size_t featureCount);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void run();
    void stop();

private:
    void handleClient(intptr_t clientSocket);
    std::string handleInfer(const std::string& body);
    std::string metricsJson() const;

    DynamicBatcher& batcher_;
    std::string address_;
    int port_;
    std::size_t featureCount_;
    intptr_t listenSocket_ = -1;
    std::atomic<bool> running_{false};
    ServerStats stats_;
};

}
