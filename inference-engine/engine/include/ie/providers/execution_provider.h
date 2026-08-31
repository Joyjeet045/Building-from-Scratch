/*
 * execution_provider.h
 * Plugin interface for the hardware backends that actually evaluate nodes,
 * modelled on ONNX Runtime's execution providers. The session owns a provider
 * and never calls a kernel directly, so adding a CUDA backend means adding a
 * subclass rather than touching the scheduling logic.
 *
 * This header deliberately declares only the interface, so callers that merely
 * hold an ExecutionProvider& do not pull in any backend's kernels.
 */
#pragma once

#include <vector>

#include "ie/core/allocator.h"
#include "ie/core/tensor.h"
#include "ie/core/thread_pool.h"
#include "ie/graph/node.h"

namespace ie {

class ExecutionProvider {
public:
    virtual ~ExecutionProvider() = default;

    virtual const char* name() const = 0;
    virtual Allocator* allocator() = 0;
    virtual bool supports(OpType op) const = 0;

    virtual ThreadPool* threadPool() { return nullptr; }

    virtual void compute(const Node& node, const std::vector<const Tensor<float>*>& inputs,
                         Tensor<float>& output) = 0;
};

}
