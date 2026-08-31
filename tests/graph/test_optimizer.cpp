/*
 * test_optimizer.cpp
 * Verifies the load-time graph rewrites: activation fusion, constant folding,
 * dead-node elimination, and that none of them change the computed result.
 */
#include <memory>
#include <unordered_map>

#include "framework/model_fixtures.h"
#include "framework/test_framework.h"
#include "ie/graph/graph_optimizer.h"
#include "ie/providers/cpu_execution_provider.h"
#include "ie/runtime/inference_engine.h"

TEST_CASE(optimizer_fuses_relu_into_the_producing_gemm) {
    ie::Model model = fixtures::buildFusableModel(false);
    ie::CpuExecutionProvider provider;
    ie::GraphOptimizer optimizer;

    ie::OptimizationReport report = optimizer.run(model, provider);

    ASSERT_EQ(report.fusedActivations, 1u);
    ASSERT_EQ(report.nodesBefore, 4u);
    ASSERT_EQ(report.nodesAfter, 3u);
    ASSERT_TRUE(model.graph().getNode("relu1") == nullptr);
    ASSERT_EQ(model.graph().getNode("fc1")->opType(), ie::OpType::GemmRelu);
    ASSERT_EQ(model.graph().getNode("fc1")->outputs().front(), std::string("hr"));
}

TEST_CASE(optimizer_eliminates_nodes_whose_output_is_never_read) {
    ie::Model model = fixtures::buildFusableModel(true);
    ie::CpuExecutionProvider provider;
    ie::GraphOptimizer optimizer;

    ie::OptimizationReport report = optimizer.run(model, provider);

    ASSERT_EQ(report.eliminatedNodes, 1u);
    ASSERT_TRUE(model.graph().getNode("unused") == nullptr);
}

TEST_CASE(optimizer_folds_nodes_whose_inputs_are_all_initializers) {
    ie::Model model;
    model.graph().addNode(fixtures::makeNode("const_add", "Add", {"a", "b"}, {"folded"}));
    model.graph().addNode(fixtures::makeNode("use", "Add", {"input", "folded"}, {"output"}));
    model.graph().setInputs({"input"});
    model.graph().setOutputs({"output"});
    model.addInitializer("a", ie::Tensor<float>({3}, {1.0f, 2.0f, 3.0f}));
    model.addInitializer("b", ie::Tensor<float>({3}, {10.0f, 20.0f, 30.0f}));

    ie::CpuExecutionProvider provider;
    ie::GraphOptimizer optimizer;
    ie::OptimizationReport report = optimizer.run(model, provider);

    ASSERT_EQ(report.foldedConstants, 1u);
    ASSERT_TRUE(model.graph().getNode("const_add") == nullptr);
    ASSERT_TRUE(model.hasInitializer("folded"));
    ASSERT_NEAR((*model.getInitializer("folded"))[2], 33.0f, 1e-6);
}

TEST_CASE(disabled_optimizer_passes_leave_the_graph_untouched) {
    ie::Model model = fixtures::buildFusableModel(true);
    ie::CpuExecutionProvider provider;

    ie::OptimizerOptions options;
    options.fuseActivations = false;
    options.foldConstants = false;
    options.eliminateDeadNodes = false;

    ie::OptimizationReport report = ie::GraphOptimizer(options).run(model, provider);

    ASSERT_EQ(report.total(), 0u);
    ASSERT_EQ(report.nodesBefore, report.nodesAfter);
    ASSERT_TRUE(model.graph().getNode("relu1") != nullptr);
}

TEST_CASE(optimization_does_not_change_results) {
    ie::Tensor<float> input({1, 4}, {1.0f, -2.0f, 0.5f, 3.0f});

    ie::SessionConfig off = ie::InferenceEngine::defaultConfig();
    off.graphOptimization = false;
    ie::InferenceEngine unoptimized(fixtures::buildFusableModel(true), off);
    ie::Tensor<float> expected = unoptimized.infer(input);

    ie::SessionConfig on = ie::InferenceEngine::defaultConfig();
    on.graphOptimization = true;
    ie::InferenceEngine optimized(fixtures::buildFusableModel(true), on);
    ie::Tensor<float> actual = optimized.infer(input);

    ASSERT_EQ(actual.shape(), expected.shape());
    for (uint64_t i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(actual[i], expected[i], 1e-5);
    }
    ASSERT_TRUE(optimized.session().optimizationReport().total() > 0u);
}
