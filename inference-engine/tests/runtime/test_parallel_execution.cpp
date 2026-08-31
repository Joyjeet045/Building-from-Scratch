/*
 * test_parallel_execution.cpp
 * Tests that running independent graph branches concurrently yields results
 * bit-identical to sequential execution, repeated enough times to catch a
 * scheduling race.
 */
#include "framework/model_fixtures.h"
#include "framework/test_framework.h"
#include "ie/runtime/inference_engine.h"

TEST_CASE(parallel_branch_execution_matches_sequential_exactly) {
    ie::Tensor<float> input({2, 4}, {0.5f, -1.0f, 2.0f, 0.25f, 1.5f, 0.0f, -0.75f, 3.0f});

    ie::SessionConfig sequential = ie::InferenceEngine::defaultConfig();
    sequential.maxBatchSize = 8;
    ie::InferenceEngine sequentialEngine(fixtures::buildBranchingModel(), sequential);
    ie::Tensor<float> expected = sequentialEngine.infer(input);

    ie::SessionConfig parallel = ie::InferenceEngine::defaultConfig();
    parallel.maxBatchSize = 8;
    parallel.intraOpThreads = 4;
    parallel.interOpParallel = true;
    ie::InferenceEngine parallelEngine(fixtures::buildBranchingModel(), parallel);

    ASSERT_TRUE(parallelEngine.session().parallelLevelCount() >= 1u);

    for (int repeat = 0; repeat < 20; ++repeat) {
        ie::Tensor<float> actual = parallelEngine.infer(input);
        ASSERT_TRUE(actual == expected);
    }
}

TEST_CASE(intra_op_threading_matches_single_threaded_results) {
    ie::Tensor<float> input({8, 4}, std::vector<float>(32, 0.75f));

    ie::SessionConfig single = ie::InferenceEngine::defaultConfig();
    single.maxBatchSize = 16;
    ie::InferenceEngine singleEngine(fixtures::buildBranchingModel(), single);
    ie::Tensor<float> expected = singleEngine.infer(input);

    ie::SessionConfig threaded = ie::InferenceEngine::defaultConfig();
    threaded.maxBatchSize = 16;
    threaded.intraOpThreads = 4;
    ie::InferenceEngine threadedEngine(fixtures::buildBranchingModel(), threaded);
    ie::Tensor<float> actual = threadedEngine.infer(input);

    ASSERT_TRUE(actual == expected);
}

TEST_CASE(session_reports_a_schedule_with_levels) {
    ie::InferenceEngine engine(fixtures::buildLinearModel(6, 4), ie::InferenceEngine::defaultConfig());

    ASSERT_TRUE(engine.session().levelCount() >= 1u);
    ASSERT_TRUE(!engine.session().describeSchedule().empty());
}
