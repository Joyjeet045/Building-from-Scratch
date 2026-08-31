/*
 * test_tensor.cpp
 * Unit tests for Tensor<T>: shape/size bookkeeping, indexing, reshape, and
 * equality.
 */
#include <cmath>

#include "framework/test_framework.h"
#include "ie/core/tensor.h"

TEST_CASE(tensor_zeros_has_expected_shape_and_size) {
    ie::Tensor<float> t = ie::Tensor<float>::zeros({2, 3});
    ASSERT_EQ(t.shape().size(), 2u);
    ASSERT_EQ(t.dim(0), 2u);
    ASSERT_EQ(t.dim(1), 3u);
    ASSERT_EQ(t.size(), 6u);
    for (uint64_t i = 0; i < t.size(); ++i) {
        ASSERT_NEAR(t[i], 0.0f, 1e-9);
    }
}

TEST_CASE(tensor_constructed_from_data_preserves_values) {
    ie::Tensor<float> t({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ASSERT_NEAR(t[0], 1.0f, 1e-9);
    ASSERT_NEAR(t[1], 2.0f, 1e-9);
    ASSERT_NEAR(t[2], 3.0f, 1e-9);
    ASSERT_NEAR(t[3], 4.0f, 1e-9);
}

TEST_CASE(tensor_constructor_rejects_mismatched_data_size) {
    ASSERT_THROWS((ie::Tensor<float>({2, 2}, std::vector<float>{1.0f, 2.0f})));
}

TEST_CASE(tensor_reshape_preserves_data) {
    ie::Tensor<float> t({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    t.reshape({3, 2});
    ASSERT_EQ(t.dim(0), 3u);
    ASSERT_EQ(t.dim(1), 2u);
    ASSERT_NEAR(t[5], 6.0f, 1e-9);
}

TEST_CASE(tensor_reshape_rejects_size_mismatch) {
    ie::Tensor<float> t({2, 3});
    ASSERT_THROWS(t.reshape({4, 4}));
}

TEST_CASE(tensor_equality_compares_shape_and_data) {
    ie::Tensor<float> a({1, 2}, {1.0f, 2.0f});
    ie::Tensor<float> b({1, 2}, {1.0f, 2.0f});
    ie::Tensor<float> c({2, 1}, {1.0f, 2.0f});
    ASSERT_TRUE(a == b);
    ASSERT_TRUE(!(a == c));
}
