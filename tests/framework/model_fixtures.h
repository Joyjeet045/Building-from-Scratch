/*
 * model_fixtures.h
 * Small hand-built models shared by the test modules, so the graph, runtime,
 * and serving suites all exercise the same known-good topologies instead of
 * each redefining their own.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ie/core/tensor.h"
#include "ie/graph/node.h"
#include "ie/model/model.h"

namespace fixtures {

inline std::unique_ptr<ie::Node> makeNode(std::string name, std::string op,
                                           std::vector<std::string> inputs,
                                           std::vector<std::string> outputs,
                                           std::unordered_map<std::string, ie::Attribute> attrs = {}) {
    return std::make_unique<ie::Node>(std::move(name), std::move(op), std::move(inputs),
                                       std::move(outputs), std::move(attrs));
}

inline std::unordered_map<std::string, ie::Attribute> gemmAttrs() {
    std::unordered_map<std::string, ie::Attribute> attrs;
    attrs.emplace("transB", ie::Attribute::makeInt(1));
    attrs.emplace("alpha", ie::Attribute::makeFloat(1.0f));
    attrs.emplace("beta", ie::Attribute::makeFloat(1.0f));
    return attrs;
}

inline std::unordered_map<std::string, ie::Attribute> flattenAttrs(int64_t axis = 1) {
    std::unordered_map<std::string, ie::Attribute> attrs;
    attrs.emplace("axis", ie::Attribute::makeInt(axis));
    return attrs;
}

inline ie::Model buildLinearModel(uint64_t features, uint64_t classes) {
    ie::Model model;
    model.graph().addNode(makeNode("flatten", "Flatten", {"input"}, {"flat"}, flattenAttrs()));
    model.graph().addNode(makeNode("fc", "Gemm", {"flat", "w", "b"}, {"output"}, gemmAttrs()));
    model.graph().setInputs({"input"});
    model.graph().setOutputs({"output"});

    std::vector<float> weights(classes * features);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = 0.01f * static_cast<float>((i * 7) % 13) - 0.05f;
    }
    model.addInitializer("w", ie::Tensor<float>({classes, features}, weights));
    model.addInitializer("b", ie::Tensor<float>({classes}, std::vector<float>(classes, 0.25f)));
    return model;
}

inline ie::Model buildBranchingModel() {
    ie::Model model;
    model.graph().addNode(makeNode("flatten", "Flatten", {"input"}, {"flat"}, flattenAttrs()));
    model.graph().addNode(makeNode("fc1", "Gemm", {"flat", "w1", "b1"}, {"h"}, gemmAttrs()));
    model.graph().addNode(makeNode("relu1", "Relu", {"h"}, {"hr"}));
    model.graph().addNode(makeNode("left", "Gemm", {"hr", "wl", "bl"}, {"left_out"}, gemmAttrs()));
    model.graph().addNode(makeNode("right", "Gemm", {"hr", "wr", "br"}, {"right_out"}, gemmAttrs()));
    model.graph().addNode(makeNode("add", "Add", {"left_out", "right_out"}, {"output"}));

    model.graph().setInputs({"input"});
    model.graph().setOutputs({"output"});

    std::vector<float> w1(16);
    for (std::size_t i = 0; i < w1.size(); ++i) w1[i] = 0.1f * static_cast<float>(i % 7) - 0.2f;
    model.addInitializer("w1", ie::Tensor<float>({4, 4}, w1));
    model.addInitializer("b1", ie::Tensor<float>({4}, {0.1f, -0.2f, 0.3f, 0.0f}));

    std::vector<float> wl(12);
    for (std::size_t i = 0; i < wl.size(); ++i) wl[i] = 0.05f * static_cast<float>(i) - 0.1f;
    model.addInitializer("wl", ie::Tensor<float>({3, 4}, wl));
    model.addInitializer("bl", ie::Tensor<float>({3}, {0.0f, 0.5f, -0.5f}));

    std::vector<float> wr(12);
    for (std::size_t i = 0; i < wr.size(); ++i) wr[i] = -0.03f * static_cast<float>(i) + 0.2f;
    model.addInitializer("wr", ie::Tensor<float>({3, 4}, wr));
    model.addInitializer("br", ie::Tensor<float>({3}, {1.0f, 0.0f, 0.25f}));

    return model;
}

inline ie::Model buildFusableModel(bool withDeadNode) {
    ie::Model model;
    model.graph().addNode(makeNode("flatten", "Flatten", {"input"}, {"flat"}, flattenAttrs()));
    model.graph().addNode(makeNode("fc1", "Gemm", {"flat", "w1", "b1"}, {"h"}, gemmAttrs()));
    model.graph().addNode(makeNode("relu1", "Relu", {"h"}, {"hr"}));
    model.graph().addNode(makeNode("fc2", "Gemm", {"hr", "w2", "b2"}, {"output"}, gemmAttrs()));
    if (withDeadNode) {
        model.graph().addNode(makeNode("unused", "Relu", {"hr"}, {"never_read"}));
    }

    model.graph().setInputs({"input"});
    model.graph().setOutputs({"output"});

    model.addInitializer("w1", ie::Tensor<float>({3, 4}, {0.5f, -0.5f, 0.25f, 1.0f, 0.1f, 0.2f, -0.3f,
                                                            0.4f, 1.0f, 0.0f, 0.0f, -1.0f}));
    model.addInitializer("b1", ie::Tensor<float>({3}, {0.1f, -2.0f, 0.0f}));
    model.addInitializer("w2", ie::Tensor<float>({2, 3}, {1.0f, 1.0f, 1.0f, 0.0f, 2.0f, -1.0f}));
    model.addInitializer("b2", ie::Tensor<float>({2}, {0.5f, -0.5f}));
    return model;
}

}
