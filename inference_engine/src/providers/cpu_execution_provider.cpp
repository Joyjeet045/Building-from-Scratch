/*
 * cpu_execution_provider.cpp
 * CPU backend: owns the intra-op thread pool and the tensor memory pool, and
 * dispatches each node to the matching kernel. GemmRelu is the fused form
 * produced by the graph optimizer and costs no extra pass over the output.
 */
#include "ie/providers/cpu_execution_provider.h"

#include <algorithm>
#include <stdexcept>

namespace ie {

CpuExecutionProvider::CpuExecutionProvider(CpuProviderOptions options) : options_(options) {
    if (options_.intraOpThreads > 1) {
        pool_ = std::make_unique<ThreadPool>(options_.intraOpThreads - 1);
    }
    if (options_.useMemoryPool) {
        poolAllocator_ = std::make_unique<PoolAllocator>();
    }
}

const char* CpuExecutionProvider::name() const { return "cpu"; }

Allocator* CpuExecutionProvider::allocator() {
    if (poolAllocator_) return poolAllocator_.get();
    return &CpuAllocator::instance();
}

bool CpuExecutionProvider::supports(OpType op) const {
    switch (op) {
        case OpType::Flatten:
        case OpType::Gemm:
        case OpType::Relu:
        case OpType::Add:
        case OpType::GemmRelu:
            return true;
        case OpType::Unknown:
            return false;
    }
    return false;
}

void CpuExecutionProvider::computeFlatten(const Node& node, const Tensor<float>& input,
                                           Tensor<float>& output) {
    int64_t axis = node.getAttrInt("axis", 1);
    const auto& shape = input.shape();
    if (axis < 0) axis += static_cast<int64_t>(shape.size());
    if (axis < 0 || static_cast<std::size_t>(axis) > shape.size()) {
        throw std::invalid_argument("Flatten axis is out of range for node " + node.name());
    }
    uint64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= shape[static_cast<std::size_t>(i)];
    if (outer == 0) outer = 1;
    uint64_t inner = input.size() / outer;

    if (output.size() != input.size()) {
        output = Tensor<float>({outer, inner}, output.allocator());
    } else {
        output.reshape({outer, inner});
    }
    std::copy(input.begin(), input.end(), output.begin());
}

void CpuExecutionProvider::computeRelu(const Tensor<float>& input, Tensor<float>& output) {
    if (output.shape() != input.shape()) {
        output = Tensor<float>(input.shape(), output.allocator());
    }
    const float* src = input.data();
    float* dst = output.data();
    for (uint64_t i = 0; i < input.size(); ++i) {
        dst[i] = std::max(0.0f, src[i]);
    }
}

void CpuExecutionProvider::computeAdd(const Tensor<float>& a, const Tensor<float>& b,
                                       Tensor<float>& output) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Add requires operands of matching shape");
    }
    if (output.shape() != a.shape()) {
        output = Tensor<float>(a.shape(), output.allocator());
    }
    const float* lhs = a.data();
    const float* rhs = b.data();
    float* dst = output.data();
    for (uint64_t i = 0; i < a.size(); ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
}

void CpuExecutionProvider::computeGemm(const Node& node, const std::vector<const Tensor<float>*>& inputs,
                                        Tensor<float>& output, bool fuseRelu) {
    if (inputs.size() < 2) {
        throw std::invalid_argument("Gemm node " + node.name() + " requires at least two inputs");
    }
    GemmParams params;
    params.transA = node.getAttrInt("transA", 0) != 0;
    params.transB = node.getAttrInt("transB", 0) != 0;
    params.alpha = node.getAttrFloat("alpha", 1.0f);
    params.beta = node.getAttrFloat("beta", 1.0f);
    params.fuseRelu = fuseRelu;

    const Tensor<float>& bias = inputs.size() > 2 ? *inputs[2] : emptyBias_;
    Gemm::compute(*inputs[0], *inputs[1], bias, output, params, options_.gemmStrategy, pool_.get());
}

void CpuExecutionProvider::compute(const Node& node, const std::vector<const Tensor<float>*>& inputs,
                                    Tensor<float>& output) {
    for (const auto* input : inputs) {
        if (input == nullptr) {
            throw std::runtime_error("Missing input tensor for node " + node.name());
        }
    }

    switch (node.opType()) {
        case OpType::Flatten:
            computeFlatten(node, *inputs.at(0), output);
            break;
        case OpType::Gemm:
            computeGemm(node, inputs, output, false);
            break;
        case OpType::GemmRelu:
            computeGemm(node, inputs, output, true);
            break;
        case OpType::Relu:
            computeRelu(*inputs.at(0), output);
            break;
        case OpType::Add:
            computeAdd(*inputs.at(0), *inputs.at(1), output);
            break;
        default:
            throw std::runtime_error("Unsupported op type '" + node.opTypeString() + "' on node " +
                                      node.name());
    }
}

}
