/*
 * test_dynamic_batcher.cpp
 * Tests that concurrently enqueued requests get coalesced into materially
 * fewer forward passes, and that malformed requests are refused.
 */
#include <future>
#include <vector>

#include "framework/model_fixtures.h"
#include "framework/test_framework.h"
#include "ie/runtime/inference_engine.h"
#include "ie/serving/dynamic_batcher.h"

TEST_CASE(dynamic_batcher_coalesces_concurrent_requests) {
    ie::SessionConfig config = ie::InferenceEngine::defaultConfig();
    config.maxBatchSize = 16;
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), config);

    ie::DynamicBatcher batcher(engine, 16, 5000, 6);

    std::vector<std::future<std::vector<float>>> futures;
    for (int i = 0; i < 24; ++i) {
        std::vector<float> features(6, 0.1f * static_cast<float>(i));
        futures.push_back(batcher.enqueue(std::move(features)));
    }

    for (auto& future : futures) {
        std::vector<float> logits = future.get();
        ASSERT_EQ(logits.size(), 4u);
    }

    ie::BatcherStats stats = batcher.stats();
    ASSERT_EQ(stats.requests, 24u);
    ASSERT_TRUE(stats.batches < 24u);
    ASSERT_TRUE(stats.maxBatchObserved > 1u);
}

TEST_CASE(dynamic_batcher_rejects_wrong_feature_count) {
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), ie::InferenceEngine::defaultConfig());
    ie::DynamicBatcher batcher(engine, 4, 1000, 6);

    ASSERT_THROWS(batcher.enqueue(std::vector<float>(5, 0.0f)));
}

TEST_CASE(dynamic_batcher_results_match_direct_inference) {
    ie::SessionConfig config = ie::InferenceEngine::defaultConfig();
    config.maxBatchSize = 8;
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), config);

    std::vector<float> features{1.0f, 0.0f, -1.0f, 0.5f, 2.0f, -0.25f};
    ie::Tensor<float> direct({1, 6}, features);
    std::vector<float> expected = engine.infer(direct).toVector();

    ie::DynamicBatcher batcher(engine, 8, 1000, 6);
    std::vector<float> actual = batcher.enqueue(features).get();

    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(actual[i], expected[i], 1e-5);
    }
}

TEST_CASE(dynamic_batcher_rejects_work_after_stop) {
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), ie::InferenceEngine::defaultConfig());
    ie::DynamicBatcher batcher(engine, 4, 1000, 6);
    batcher.stop();

    ASSERT_THROWS(batcher.enqueue(std::vector<float>(6, 0.0f)));
}
