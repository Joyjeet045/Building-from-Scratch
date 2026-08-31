/*
 * cpu_execution_provider.h
 * The CPU backend. Owns the worker pool used for intra-op and inter-op
 * parallelism and the arena the session allocates tensors from.
 */
#pragma once

#include <cstddef>
#include <memory>

#include "ie/kernels/gemm.h"
#include "ie/providers/execution_provider.h"

namespace ie {

struct CpuProviderOptions {
    GemmStrategy gemmStrategy = GemmStrategy::Auto;
    std::size_t intraOpThreads = 1;
    bool useMemoryPool = true;
};

class CpuExecutionProvider final : public ExecutionProvider {
public:
    explicit CpuExecutionProvider(CpuProviderOptions options = {});

    const char* name() const override;
    Allocator* allocator() override;
    bool supports(OpType op) const override;

    void compute(const Node& node, const std::vector<const Tensor<float>*>& inputs,
                 Tensor<float>& output) override;

    ThreadPool* threadPool() override { return pool_.get(); }
    const PoolAllocator* pool() const { return poolAllocator_.get(); }
    GemmStrategy gemmStrategy() const { return options_.gemmStrategy; }
    void setGemmStrategy(GemmStrategy strategy) { options_.gemmStrategy = strategy; }

private:
    void computeGemm(const Node& node, const std::vector<const Tensor<float>*>& inputs,
                     Tensor<float>& output, bool fuseRelu);
    void computeFlatten(const Node& node, const Tensor<float>& input, Tensor<float>& output);
    void computeRelu(const Tensor<float>& input, Tensor<float>& output);
    void computeAdd(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& output);

    CpuProviderOptions options_;
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<PoolAllocator> poolAllocator_;
    Tensor<float> emptyBias_;
};

}
