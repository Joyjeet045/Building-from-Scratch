/*
 * inference_engine.cpp
 * Thin facade implementation delegating to InferenceSession.
 */
#include "ie/runtime/inference_engine.h"

#include <stdexcept>

namespace ie {

SessionConfig InferenceEngine::defaultConfig() {
    SessionConfig config;
    config.modelPath = "<in-memory>";
    return config;
}

InferenceEngine::InferenceEngine(Model model, SessionConfig config)
    : session_(std::make_unique<InferenceSession>(std::move(model), std::move(config))) {}

InferenceEngine::InferenceEngine(const SessionConfig& config)
    : session_(InferenceSession::fromConfig(config)) {}

Tensor<float> InferenceEngine::infer(const Tensor<float>& input) { return session_->run(input); }

const Tensor<float>& InferenceEngine::inferBatch(const Tensor<float>& batch) {
    if (batch.rank() == 0) {
        throw std::invalid_argument("Batch tensor must have at least one dimension");
    }
    std::size_t maxBatch = session_->config().maxBatchSize;
    if (batch.dim(0) > maxBatch) {
        throw std::invalid_argument("Batch size " + std::to_string(batch.dim(0)) +
                                     " exceeds max_batch_size " + std::to_string(maxBatch));
    }
    return session_->run(batch);
}

}
