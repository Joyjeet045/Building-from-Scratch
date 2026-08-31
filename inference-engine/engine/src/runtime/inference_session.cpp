/*
 * inference_session.cpp
 * Implementation of the session: load-time optimization, tensor store
 * pre-allocation, and the sequential and wavefront-parallel schedulers.
 *
 * Thread safety of the parallel path rests on two invariants established in
 * prepare(): every tensor slot exists before execution starts, so the map
 * never rehashes while workers hold references into it, and each node in a
 * level owns a distinct output name, so no two workers touch the same slot.
 */
#include "ie/runtime/inference_session.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "ie/model/model_loader.h"
#include "ie/providers/provider_factory.h"

namespace ie {

namespace {

thread_local std::vector<const Tensor<float>*> tInputScratch;

}

InferenceSession::InferenceSession(Model model, SessionConfig config)
    : config_(std::move(config)), model_(std::move(model)) {
    config_.validate();
    provider_ = makeExecutionProvider(config_.executionProvider, config_.cpuOptions());
    profiler_.setEnabled(config_.profiling);
    prepare();
}

InferenceSession::~InferenceSession() {
    tensors_.clear();
}

std::unique_ptr<InferenceSession> InferenceSession::fromConfig(const SessionConfig& config) {
    config.validate();
    Model model = ModelLoader::load(config.modelPath);
    return std::make_unique<InferenceSession>(std::move(model), config);
}

void InferenceSession::prepare() {
    if (model_.graph().inputs().empty()) {
        throw std::runtime_error("Model graph declares no inputs");
    }
    if (model_.graph().outputs().empty()) {
        throw std::runtime_error("Model graph declares no outputs");
    }

    if (config_.graphOptimization) {
        GraphOptimizer optimizer(config_.optimizerOptions());
        optimizationReport_ = optimizer.run(model_, *provider_);
    } else {
        optimizationReport_.nodesBefore = model_.graph().size();
        optimizationReport_.nodesAfter = model_.graph().size();
    }

    inputName_ = model_.graph().inputs().front();
    outputName_ = model_.graph().outputs().front();

    order_ = model_.graph().topologicalSort();
    levels_ = model_.graph().executionLevels();

    for (Node* node : order_) {
        if (!provider_->supports(node->opType())) {
            throw std::runtime_error("Execution provider '" + std::string(provider_->name()) +
                                      "' does not support op '" + node->opTypeString() + "' on node " +
                                      node->name());
        }
        if (node->outputs().size() != 1) {
            throw std::runtime_error("Node " + node->name() + " must produce exactly one output");
        }
    }

    Allocator* allocator = provider_->allocator();

    std::size_t slots = model_.initializers().size() + order_.size() + 2;
    tensors_.reserve(slots * 2);

    for (const auto& [name, tensor] : model_.initializers()) {
        Tensor<float> resident(tensor.shape(), allocator);
        std::copy(tensor.begin(), tensor.end(), resident.begin());
        tensors_.emplace(name, std::move(resident));
    }

    tensors_.emplace(inputName_, Tensor<float>({}, allocator));
    for (Node* node : order_) {
        tensors_.emplace(node->outputs().front(), Tensor<float>({}, allocator));
    }

    if (tensors_.find(outputName_) == tensors_.end()) {
        throw std::runtime_error("Graph output '" + outputName_ + "' is not produced by any node");
    }

    for (Node* node : order_) {
        for (const auto& input : node->inputs()) {
            if (tensors_.find(input) == tensors_.end()) {
                throw std::runtime_error("Node " + node->name() + " reads unknown tensor '" + input + "'");
            }
        }
    }
}

void InferenceSession::executeNode(Node* node) {
    std::vector<const Tensor<float>*>& inputs = tInputScratch;
    inputs.clear();
    inputs.reserve(node->inputs().size());
    for (const auto& name : node->inputs()) {
        inputs.push_back(&tensors_.find(name)->second);
    }

    Tensor<float>& output = tensors_.find(node->outputs().front())->second;

    if (profiler_.enabled()) {
        ScopedTimer timer;
        provider_->compute(*node, inputs, output);
        profiler_.record(node->name(), node->opTypeString(), timer.elapsedMs());
    } else {
        provider_->compute(*node, inputs, output);
    }
}

const Tensor<float>& InferenceSession::run(const Tensor<float>& input) {
    if (input.size() == 0) {
        throw std::invalid_argument("Inference input tensor is empty");
    }

    ScopedTimer wall;

    Tensor<float>& slot = tensors_.find(inputName_)->second;
    if (slot.shape() != input.shape()) {
        slot = Tensor<float>(input.shape(), provider_->allocator());
    }
    std::copy(input.begin(), input.end(), slot.begin());

    ThreadPool* pool = provider_->threadPool();
    bool parallel = config_.interOpParallel && pool != nullptr;

    if (parallel) {
        std::vector<std::function<void()>> tasks;
        for (const auto& level : levels_) {
            if (level.size() == 1) {
                executeNode(level.front());
                continue;
            }
            tasks.clear();
            tasks.reserve(level.size());
            for (Node* node : level) {
                tasks.emplace_back([this, node] { executeNode(node); });
            }
            pool->runAll(tasks);
        }
    } else {
        for (Node* node : order_) {
            executeNode(node);
        }
    }

    profiler_.recordTotal(wall.elapsedMs());
    return tensors_.find(outputName_)->second;
}

std::size_t InferenceSession::parallelLevelCount() const {
    std::size_t count = 0;
    for (const auto& level : levels_) {
        if (level.size() > 1) ++count;
    }
    return count;
}

std::string InferenceSession::describeSchedule() const {
    std::ostringstream oss;
    for (std::size_t i = 0; i < levels_.size(); ++i) {
        oss << "  level " << i << (levels_[i].size() > 1 ? " [parallel] " : "            ");
        for (std::size_t j = 0; j < levels_[i].size(); ++j) {
            oss << levels_[i][j]->name() << "(" << levels_[i][j]->opTypeString() << ")";
            if (j + 1 < levels_[i].size()) oss << ", ";
        }
        oss << "\n";
    }
    return oss.str();
}

}
