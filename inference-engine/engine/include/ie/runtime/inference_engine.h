/*
 * inference_engine.h
 * Public facade over InferenceSession. Adds batch handling on top: a batch of
 * N inputs is a single (N, ...) tensor, so one forward pass serves N requests
 * and amortizes the per-call overhead across all of them.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ie/core/tensor.h"
#include "ie/model/model.h"
#include "ie/runtime/config.h"
#include "ie/runtime/inference_session.h"

namespace ie {

class InferenceEngine {
public:
    explicit InferenceEngine(Model model, SessionConfig config = defaultConfig());
    explicit InferenceEngine(const SessionConfig& config);

    static SessionConfig defaultConfig();

    Tensor<float> infer(const Tensor<float>& input);
    const Tensor<float>& inferBatch(const Tensor<float>& batch);

    InferenceSession& session() { return *session_; }
    const InferenceSession& session() const { return *session_; }

private:
    std::unique_ptr<InferenceSession> session_;
};

}
