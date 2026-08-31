/*
 * dynamic_batcher.cpp
 * Implementation of the queue-and-coalesce worker. Exceptions raised while
 * running a batch are forwarded to every request in that batch so no caller
 * is left waiting on a promise that will never be satisfied.
 */
#include "ie/serving/dynamic_batcher.h"

#include <chrono>
#include <stdexcept>

#include "ie/runtime/profiler.h"

namespace ie {

DynamicBatcher::DynamicBatcher(InferenceEngine& engine, std::size_t maxBatchSize,
                                std::size_t maxQueueDelayUs, std::size_t featureCount)
    : engine_(engine),
      maxBatchSize_(maxBatchSize == 0 ? 1 : maxBatchSize),
      maxQueueDelayUs_(maxQueueDelayUs),
      featureCount_(featureCount) {
    worker_ = std::thread([this] { workerLoop(); });
}

DynamicBatcher::~DynamicBatcher() { stop(); }

void DynamicBatcher::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return;
        stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::future<std::vector<float>> DynamicBatcher::enqueue(std::vector<float> features) {
    if (features.size() != featureCount_) {
        throw std::invalid_argument("Expected " + std::to_string(featureCount_) + " features, got " +
                                     std::to_string(features.size()));
    }

    Request request;
    request.features = std::move(features);
    std::future<std::vector<float>> future = request.promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) throw std::runtime_error("Batcher is shutting down");
        queue_.push_back(std::move(request));
    }
    condition_.notify_one();
    return future;
}

void DynamicBatcher::runBatch(std::vector<Request>& batch) {
    try {
        uint64_t n = static_cast<uint64_t>(batch.size());
        uint64_t features = static_cast<uint64_t>(featureCount_);

        Tensor<float> input({n, features});
        float* dst = input.data();
        for (std::size_t i = 0; i < batch.size(); ++i) {
            std::copy(batch[i].features.begin(), batch[i].features.end(), dst + i * features);
        }

        ScopedTimer timer;
        const Tensor<float>& output = engine_.inferBatch(input);
        double elapsed = timer.elapsedMs();

        if (output.rank() != 2 || output.dim(0) != n) {
            throw std::runtime_error("Batched output has unexpected shape");
        }
        uint64_t classes = output.dim(1);
        const float* src = output.data();

        for (std::size_t i = 0; i < batch.size(); ++i) {
            std::vector<float> logits(src + i * classes, src + (i + 1) * classes);
            batch[i].promise.set_value(std::move(logits));
        }

        std::lock_guard<std::mutex> lock(mutex_);
        stats_.requests += batch.size();
        ++stats_.batches;
        stats_.totalBatchMs += elapsed;
        if (batch.size() > stats_.maxBatchObserved) stats_.maxBatchObserved = batch.size();
    } catch (...) {
        std::exception_ptr error = std::current_exception();
        for (auto& request : batch) {
            request.promise.set_exception(error);
        }
    }
}

void DynamicBatcher::workerLoop() {
    for (;;) {
        std::vector<Request> batch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) return;

            if (queue_.size() < maxBatchSize_ && maxQueueDelayUs_ > 0) {
                condition_.wait_for(lock, std::chrono::microseconds(maxQueueDelayUs_), [this] {
                    return stopping_ || queue_.size() >= maxBatchSize_;
                });
            }

            std::size_t take = queue_.size() < maxBatchSize_ ? queue_.size() : maxBatchSize_;
            batch.reserve(take);
            for (std::size_t i = 0; i < take; ++i) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }

        if (!batch.empty()) runBatch(batch);
    }
}

BatcherStats DynamicBatcher::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}
