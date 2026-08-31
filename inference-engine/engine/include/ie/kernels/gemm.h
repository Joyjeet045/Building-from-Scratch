/*
 * gemm.h
 * Generalized matrix multiply kernels, one per optimization step from the
 * matrix-multiplication write-up, kept side by side so they can be
 * benchmarked against each other:
 *   Naive     - textbook i,j,k triple loop
 *   Reordered - i,k,j so the inner loop walks memory contiguously
 *   Tiled     - blocked to keep the working set inside cache
 *   Packed    - blocked plus tile packing to avoid cache-set conflicts
 *   Auto      - picks between Tiled and Packed at the measured crossover
 *               (batch >= 16, where packing overhead starts paying off)
 * Every kernel computes C = alpha * op(A) * op(B) + beta * bias, where bias
 * is broadcast per column, and all of them optionally shard output rows
 * across a ThreadPool. The blocked kernels skip zero multipliers, which on
 * MNIST inputs is a large win because the images are mostly black.
 */
#pragma once

#include <cstdint>
#include <string>

#include "ie/core/tensor.h"
#include "ie/core/thread_pool.h"

namespace ie {

enum class GemmStrategy { Naive, Reordered, Tiled, Packed, Auto };

GemmStrategy gemmStrategyFromString(const std::string& name);
const char* gemmStrategyName(GemmStrategy strategy);

struct GemmParams {
    bool transA = false;
    bool transB = false;
    float alpha = 1.0f;
    float beta = 1.0f;
    bool fuseRelu = false;
};

class Gemm {
public:
    static void compute(const Tensor<float>& A, const Tensor<float>& B, const Tensor<float>& bias,
                        Tensor<float>& out, const GemmParams& params, GemmStrategy strategy,
                        ThreadPool* pool);

    static void validate(const Tensor<float>& A, const Tensor<float>& B, const Tensor<float>& bias,
                         const GemmParams& params);

    static std::vector<uint64_t> outputShape(const Tensor<float>& A, const Tensor<float>& B,
                                              const GemmParams& params);
};

}
