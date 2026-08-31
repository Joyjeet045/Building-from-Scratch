/*
 * test_inference_engine.cpp
 * End-to-end test of InferenceEngine against a small hand-built model
 * (Flatten -> Gemm -> Relu -> Gemm) with known weights, verifying the whole
 * load-free execution path produces the expected logits.
 */
#include <memory>
#include <unordered_map>

#include "framework/test_framework.h"
#include "ie/model/model.h"
#include "ie/runtime/inference_engine.h"

TEST_CASE(inference_engine_runs_branchless_mlp_end_to_end) {
    ie::Model model;

    std::unordered_map<std::string, ie::Attribute> flattenAttrs;
    flattenAttrs.emplace("axis", ie::Attribute::makeInt(1));
    model.graph().addNode(
        std::make_unique<ie::Node>("flatten", "Flatten", std::vector<std::string>{"input"},
                                    std::vector<std::string>{"flat"}, flattenAttrs));

    std::unordered_map<std::string, ie::Attribute> gemmAttrs;
    gemmAttrs.emplace("transB", ie::Attribute::makeInt(1));
    gemmAttrs.emplace("alpha", ie::Attribute::makeFloat(1.0f));
    gemmAttrs.emplace("beta", ie::Attribute::makeFloat(1.0f));

    model.graph().addNode(std::make_unique<ie::Node>(
        "fc1", "Gemm", std::vector<std::string>{"flat", "w1", "b1"}, std::vector<std::string>{"h"}, gemmAttrs));

    model.graph().addNode(std::make_unique<ie::Node>("relu1", "Relu", std::vector<std::string>{"h"},
                                                       std::vector<std::string>{"hr"},
                                                       std::unordered_map<std::string, ie::Attribute>{}));

    model.graph().addNode(std::make_unique<ie::Node>(
        "fc2", "Gemm", std::vector<std::string>{"hr", "w2", "b2"}, std::vector<std::string>{"output"}, gemmAttrs));

    model.graph().setInputs({"input"});
    model.graph().setOutputs({"output"});

    model.addInitializer("w1", ie::Tensor<float>({3, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}));
    model.addInitializer("b1", ie::Tensor<float>({3}, {0, 1, -5}));
    model.addInitializer("w2", ie::Tensor<float>({2, 3}, {1, 1, 1, 2, 0, 0}));
    model.addInitializer("b2", ie::Tensor<float>({2}, {0, 1}));

    ie::InferenceEngine engine(std::move(model));

    ie::Tensor<float> input({1, 2, 2}, {1, 2, 3, 4});
    ie::Tensor<float> output = engine.infer(input);

    ASSERT_EQ(output.shape().size(), 2u);
    ASSERT_EQ(output.dim(0), 1u);
    ASSERT_EQ(output.dim(1), 2u);
    ASSERT_NEAR(output[0], 4.0f, 1e-5);
    ASSERT_NEAR(output[1], 3.0f, 1e-5);
}