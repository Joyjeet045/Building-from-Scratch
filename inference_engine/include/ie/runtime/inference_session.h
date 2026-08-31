/*
 * inference_session.h
 * Owns a model, an execution provider, and the tensor store, and drives one
 * inference. Inputs for a node are gathered as raw pointers rather than copies
 * - the single change that dominated the engine's original speedup - and
 * output buffers are allocated once and reused across runs.
 *
 * Scheduling has two modes. Sequential walks the topological order. Parallel
 * walks wavefront levels: nodes in a level have no dependency on each other,
 * so branches such as the model's two parallel Gemm paths execute
 * concurrently. Levels holding a single node instead fall back to intra-op
 * parallelism inside that node's kernel.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ie/core/tensor.h"
#include "ie/graph/graph_optimizer.h"
#include "ie/model/model.h"
#include "ie/providers/execution_provider.h"
#include "ie/runtime/config.h"
#include "ie/runtime/profiler.h"

namespace ie {

class InferenceSession {
public:
    InferenceSession(Model model, SessionConfig config);
    ~InferenceSession();

    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    static std::unique_ptr<InferenceSession> fromConfig(const SessionConfig& config);

    const Tensor<float>& run(const Tensor<float>& input);

    const SessionConfig& config() const { return config_; }
    const OptimizationReport& optimizationReport() const { return optimizationReport_; }
    ExecutionProvider& provider() { return *provider_; }
    Profiler& profiler() { return profiler_; }
    const Profiler& profiler() const { return profiler_; }

    std::size_t levelCount() const { return levels_.size(); }
    std::size_t parallelLevelCount() const;
    std::string describeSchedule() const;

private:
    void prepare();
    void executeNode(Node* node);

    SessionConfig config_;
    Model model_;
    std::unique_ptr<ExecutionProvider> provider_;

    std::vector<Node*> order_;
    std::vector<std::vector<Node*>> levels_;
    std::unordered_map<std::string, Tensor<float>> tensors_;

    OptimizationReport optimizationReport_;
    Profiler profiler_;
    std::string inputName_;
    std::string outputName_;
};

}
