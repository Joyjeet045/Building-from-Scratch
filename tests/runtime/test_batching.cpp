/*
 * test_batching.cpp
 * Tests that a batched forward pass produces the same logits as running each
 * sample individually, and that batch bounds are enforced.
 */
#include <vector>

#include "framework/model_fixtures.h"
#include "framework/test_framework.h"
#include "ie/runtime/inference_engine.h"

TEST_CASE(batched_inference_matches_per_sample_inference) {
    ie::SessionConfig config = ie::InferenceEngine::defaultConfig();
    config.maxBatchSize = 8;
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), config);

    std::vector<std::vector<float>> samples{{1.0f, 0.0f, -1.0f, 0.5f, 2.0f, -0.25f},
                                             {0.0f, 3.0f, 1.5f, -2.0f, 0.25f, 1.0f},
                                             {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f}};

    std::vector<std::vector<float>> individual;
    for (const auto& sample : samples) {
        ie::Tensor<float> input({1, 6}, sample);
        individual.push_back(engine.infer(input).toVector());
    }

    std::vector<float> flat;
    for (const auto& sample : samples) flat.insert(flat.end(), sample.begin(), sample.end());
    ie::Tensor<float> batch({3, 6}, flat);
    const ie::Tensor<float>& batched = engine.inferBatch(batch);

    ASSERT_EQ(batched.dim(0), 3u);
    ASSERT_EQ(batched.dim(1), 4u);
    for (std::size_t s = 0; s < samples.size(); ++s) {
        for (uint64_t c = 0; c < 4; ++c) {
            ASSERT_NEAR(batched[s * 4 + c], individual[s][c], 1e-4);
        }
    }
}

TEST_CASE(batch_larger_than_max_batch_size_is_rejected) {
    ie::SessionConfig config = ie::InferenceEngine::defaultConfig();
    config.maxBatchSize = 2;
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), config);

    ie::Tensor<float> batch({3, 6}, std::vector<float>(18, 1.0f));
    ASSERT_THROWS(engine.inferBatch(batch));
}

TEST_CASE(empty_input_is_rejected) {
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), ie::InferenceEngine::defaultConfig());
    ie::Tensor<float> empty;
    ASSERT_THROWS(engine.infer(empty));
}

TEST_CASE(changing_batch_size_between_runs_is_supported) {
    ie::SessionConfig config = ie::InferenceEngine::defaultConfig();
    config.maxBatchSize = 8;
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), config);

    for (uint64_t n : {1u, 4u, 2u, 8u}) {
        ie::Tensor<float> batch({n, 6}, std::vector<float>(n * 6, 0.5f));
        const ie::Tensor<float>& out = engine.inferBatch(batch);
        ASSERT_EQ(out.dim(0), n);
        ASSERT_EQ(out.dim(1), 4u);
    }
}
