/*
 * operators.cpp
 * Straightforward reference implementations of the supported operators. The
 * execution provider uses the optimized kernels instead; these exist as the
 * unambiguous definition of correct behaviour that tests check the fast paths
 * against.
 */
#include "ie/kernels/operators.h"

#include <algorithm>
#include <stdexcept>

#include "ie/kernels/gemm.h"

namespace ie {

template <typename T>
Tensor<T> CpuOperators<T>::flatten(const Tensor<T>& input, int64_t axis) {
    const auto& shape = input.shape();
    if (axis < 0) axis += static_cast<int64_t>(shape.size());
    if (axis < 0 || static_cast<std::size_t>(axis) > shape.size()) {
        throw std::invalid_argument("Flatten axis is out of range");
    }
    uint64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= shape[static_cast<std::size_t>(i)];
    if (outer == 0) outer = 1;
    uint64_t inner = input.size() / outer;
    Tensor<T> output = input;
    output.reshape({outer, inner});
    return output;
}

template <typename T>
Tensor<T> CpuOperators<T>::relu(const Tensor<T>& input) {
    Tensor<T> output = input;
    for (uint64_t i = 0; i < output.size(); ++i) {
        output[i] = std::max(static_cast<T>(0), output[i]);
    }
    return output;
}

template <typename T>
Tensor<T> CpuOperators<T>::add(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.shape() != B.shape()) {
        throw std::invalid_argument("Add requires operands of matching shape");
    }
    Tensor<T> output(A.shape());
    for (uint64_t i = 0; i < output.size(); ++i) {
        output[i] = A[i] + B[i];
    }
    return output;
}

template <>
Tensor<float> CpuOperators<float>::gemm(const Tensor<float>& A, const Tensor<float>& B,
                                         const Tensor<float>& bias, bool transA, bool transB, float alpha,
                                         float beta) {
    GemmParams params;
    params.transA = transA;
    params.transB = transB;
    params.alpha = alpha;
    params.beta = beta;

    Tensor<float> output(Gemm::outputShape(A, B, params));
    Gemm::compute(A, B, bias, output, params, GemmStrategy::Naive, nullptr);
    return output;
}

template class CpuOperators<float>;

}
