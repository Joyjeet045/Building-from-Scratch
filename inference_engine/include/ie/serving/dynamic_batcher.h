/*
 * dynamic_batcher.h
 * Server-side throughput optimization: incoming requests are queued rather
 * than executed immediately, and a single worker forms them into one batched
 * forward pass once either max_batch_size requests have arrived or
 * max_queue_delay_microseconds has elapsed. This trades a bounded amount of
 * latency for better hardware utilization, the same tradeoff production
 * inference servers make.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "ie/runtime/inference_engine.h"

namespace ie {

struct BatcherStats {
    uint64_t requests = 0;
    uint64_t batches = 0;
    uint64_t maxBatchObserved = 0;
    double totalBatchMs = 0.0;

    double averageBatchSize() const {
        return batches == 0 ? 0.0 : static_cast<double>(requests) / static_cast<double>(batches);
    }
};

class DynamicBatcher {
public:
    DynamicBatcher(InferenceEngine& engine, std::size_t maxBatchSize, std::size_t maxQueueDelayUs,
                   std::size_t featureCount);
    ~DynamicBatcher();

    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    std::future<std::vector<float>> enqueue(std::vector<float> features);

    BatcherStats stats() const;
    void stop();

private:
    struct Request {
        std::vector<float> features;
        std::promise<std::vector<float>> promise;
    };

    void workerLoop();
    void runBatch(std::vector<Request>& batch);

    InferenceEngine& engine_;
    std::size_t maxBatchSize_;
    std::size_t maxQueueDelayUs_;
    std::size_t featureCount_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Request> queue_;
    bool stopping_ = false;
    std::thread worker_;
    BatcherStats stats_;
};

}
