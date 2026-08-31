/*
 * operators.h
 * Declares the CPU kernels backing each supported op type. Kept as a small
 * templated class so future execution providers (e.g. a GPU backend) can
 * implement the same interface.
 */
#pragma once

#include "ie/core/tensor.h"

namespace ie {

template <typename T>
class CpuOperators {
public:
    static Tensor<T> flatten(const Tensor<T>& input, int64_t axis);
    static Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& bias, bool transA,
                          bool transB, float alpha, float beta);
    static Tensor<T> relu(const Tensor<T>& input);
    static Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B);
};

}
