/*
 * test_gemm.cpp
 * Checks every optimized GEMM strategy against the naive reference on shapes
 * that exercise the tiling and packing boundaries, plus the fused-ReLU path
 * and the thread-sharded path.
 */
#include <cmath>
#include <random>
#include <vector>

#include "framework/test_framework.h"
#include "ie/kernels/gemm.h"

namespace {

ie::Tensor<float> randomTensor(std::vector<uint64_t> shape, unsigned seed) {
    uint64_t total = 1;
    for (uint64_t d : shape) total *= d;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(total);
    for (auto& value : values) value = dist(rng);
    return ie::Tensor<float>(std::move(shape), values);
}

void expectMatchesReference(uint64_t N, uint64_t K, uint64_t M, bool transB, ie::GemmStrategy strategy,
                             ie::ThreadPool* pool) {
    ie::GemmParams params;
    params.transB = transB;

    ie::Tensor<float> A = randomTensor({N, K}, 1);
    ie::Tensor<float> B = transB ? randomTensor({M, K}, 2) : randomTensor({K, M}, 2);
    ie::Tensor<float> bias = randomTensor({M}, 3);

    ie::Tensor<float> reference({N, M});
    ie::Gemm::compute(A, B, bias, reference, params, ie::GemmStrategy::Naive, nullptr);

    ie::Tensor<float> actual({N, M});
    ie::Gemm::compute(A, B, bias, actual, params, strategy, pool);

    for (uint64_t i = 0; i < N * M; ++i) {
        ASSERT_NEAR(actual[i], reference[i], 1e-3);
    }
}

}

TEST_CASE(gemm_reordered_matches_naive) {
    expectMatchesReference(7, 33, 19, false, ie::GemmStrategy::Reordered, nullptr);
    expectMatchesReference(7, 33, 19, true, ie::GemmStrategy::Reordered, nullptr);
}

TEST_CASE(gemm_tiled_matches_naive_across_tile_boundaries) {
    expectMatchesReference(65, 130, 67, false, ie::GemmStrategy::Tiled, nullptr);
    expectMatchesReference(65, 130, 67, true, ie::GemmStrategy::Tiled, nullptr);
}

TEST_CASE(gemm_packed_matches_naive_across_tile_boundaries) {
    expectMatchesReference(65, 130, 67, false, ie::GemmStrategy::Packed, nullptr);
    expectMatchesReference(65, 130, 67, true, ie::GemmStrategy::Packed, nullptr);
}

TEST_CASE(gemm_auto_matches_naive) {
    expectMatchesReference(32, 128, 64, true, ie::GemmStrategy::Auto, nullptr);
}

TEST_CASE(gemm_threaded_matches_single_threaded) {
    ie::ThreadPool pool(3);
    expectMatchesReference(96, 200, 80, true, ie::GemmStrategy::Packed, &pool);
    expectMatchesReference(96, 200, 80, false, ie::GemmStrategy::Tiled, &pool);
}

TEST_CASE(gemm_fused_relu_equals_gemm_then_relu) {
    ie::GemmParams plain;
    plain.transB = true;
    ie::GemmParams fused = plain;
    fused.fuseRelu = true;

    ie::Tensor<float> A = randomTensor({9, 40}, 11);
    ie::Tensor<float> B = randomTensor({15, 40}, 12);
    ie::Tensor<float> bias = randomTensor({15}, 13);

    ie::Tensor<float> unfused({9, 15});
    ie::Gemm::compute(A, B, bias, unfused, plain, ie::GemmStrategy::Auto, nullptr);

    ie::Tensor<float> withRelu({9, 15});
    ie::Gemm::compute(A, B, bias, withRelu, fused, ie::GemmStrategy::Auto, nullptr);

    for (uint64_t i = 0; i < unfused.size(); ++i) {
        ASSERT_NEAR(withRelu[i], std::max(0.0f, unfused[i]), 1e-6);
    }
}

TEST_CASE(gemm_rejects_mismatched_inner_dimensions) {
    ie::GemmParams params;
    ie::Tensor<float> A({2, 3});
    ie::Tensor<float> B({4, 5});
    ie::Tensor<float> bias({5});
    ASSERT_THROWS(ie::Gemm::validate(A, B, bias, params));
}

TEST_CASE(gemm_strategy_names_round_trip) {
    for (const char* name : {"naive", "reordered", "tiled", "packed", "auto"}) {
        ASSERT_EQ(std::string(ie::gemmStrategyName(ie::gemmStrategyFromString(name))), std::string(name));
    }
    ASSERT_THROWS(ie::gemmStrategyFromString("nonexistent"));
}
