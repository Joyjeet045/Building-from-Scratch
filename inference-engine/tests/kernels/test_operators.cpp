/*
 * test_operators.cpp
 * Unit tests for the CPU kernels: flatten, gemm (with/without transpose),
 * relu, and add.
 */
#include "framework/test_framework.h"
#include "ie/kernels/operators.h"

TEST_CASE(flatten_collapses_dims_after_axis) {
    ie::Tensor<float> input({1, 2, 3}, {1, 2, 3, 4, 5, 6});
    ie::Tensor<float> output = ie::CpuOperators<float>::flatten(input, 1);
    ASSERT_EQ(output.shape().size(), 2u);
    ASSERT_EQ(output.dim(0), 1u);
    ASSERT_EQ(output.dim(1), 6u);
    ASSERT_NEAR(output[4], 5.0f, 1e-9);
}

TEST_CASE(gemm_computes_matrix_product_with_bias) {
    ie::Tensor<float> A({2, 2}, {1, 2, 3, 4});
    ie::Tensor<float> B({2, 2}, {5, 6, 7, 8});
    ie::Tensor<float> bias({2}, {1, 1});
    ie::Tensor<float> output = ie::CpuOperators<float>::gemm(A, B, bias, false, false, 1.0f, 1.0f);
    ASSERT_NEAR(output[0], 19 + 1, 1e-6);
    ASSERT_NEAR(output[1], 22 + 1, 1e-6);
    ASSERT_NEAR(output[2], 43 + 1, 1e-6);
    ASSERT_NEAR(output[3], 50 + 1, 1e-6);
}

TEST_CASE(gemm_honors_transB_like_torch_linear) {
    ie::Tensor<float> A({1, 2}, {1, 2});
    ie::Tensor<float> weight({3, 2}, {1, 0, 0, 1, 1, 1});
    ie::Tensor<float> bias({3}, {0, 0, 0});
    ie::Tensor<float> output = ie::CpuOperators<float>::gemm(A, weight, bias, false, true, 1.0f, 1.0f);
    ASSERT_EQ(output.dim(0), 1u);
    ASSERT_EQ(output.dim(1), 3u);
    ASSERT_NEAR(output[0], 1.0f, 1e-6);
    ASSERT_NEAR(output[1], 2.0f, 1e-6);
    ASSERT_NEAR(output[2], 3.0f, 1e-6);
}

TEST_CASE(relu_clamps_negative_values_to_zero) {
    ie::Tensor<float> input({4}, {-1.0f, 0.0f, 2.0f, -3.5f});
    ie::Tensor<float> output = ie::CpuOperators<float>::relu(input);
    ASSERT_NEAR(output[0], 0.0f, 1e-9);
    ASSERT_NEAR(output[1], 0.0f, 1e-9);
    ASSERT_NEAR(output[2], 2.0f, 1e-9);
    ASSERT_NEAR(output[3], 0.0f, 1e-9);
}

TEST_CASE(add_sums_matching_shapes_elementwise) {
    ie::Tensor<float> A({3}, {1.0f, 2.0f, 3.0f});
    ie::Tensor<float> B({3}, {10.0f, 20.0f, 30.0f});
    ie::Tensor<float> output = ie::CpuOperators<float>::add(A, B);
    ASSERT_NEAR(output[0], 11.0f, 1e-9);
    ASSERT_NEAR(output[1], 22.0f, 1e-9);
    ASSERT_NEAR(output[2], 33.0f, 1e-9);
}

TEST_CASE(add_rejects_mismatched_shapes) {
    ie::Tensor<float> A({2}, {1.0f, 2.0f});
    ie::Tensor<float> B({3}, {1.0f, 2.0f, 3.0f});
    ASSERT_THROWS(ie::CpuOperators<float>::add(A, B));
}
