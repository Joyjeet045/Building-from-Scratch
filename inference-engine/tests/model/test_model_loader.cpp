/*
 * test_model_loader.cpp
 * Round-trips a hand-built Model through ModelLoader::save/load and checks
 * the reloaded graph, initializers and inference output all match.
 */
#include <filesystem>
#include <memory>
#include <unordered_map>

#include "framework/test_framework.h"
#include "ie/model/model_loader.h"
#include "ie/runtime/inference_engine.h"

namespace {

ie::Model buildSampleModel() {
    ie::Model model;

    std::unordered_map<std::string, ie::Attribute> gemmAttrs;
    gemmAttrs.emplace("transB", ie::Attribute::makeInt(1));
    gemmAttrs.emplace("alpha", ie::Attribute::makeFloat(1.0f));
    gemmAttrs.emplace("beta", ie::Attribute::makeFloat(1.0f));

    model.graph().addNode(std::make_unique<ie::Node>(
        "fc1", "Gemm", std::vector<std::string>{"input", "w1", "b1"}, std::vector<std::string>{"h"}, gemmAttrs));
    model.graph().addNode(std::make_unique<ie::Node>("relu1", "Relu", std::vector<std::string>{"h"},
                                                       std::vector<std::string>{"output"},
                                                       std::unordered_map<std::string, ie::Attribute>{}));

    model.graph().setInputs({"input"});
    model.graph().setOutputs({"output"});

    model.addInitializer("w1", ie::Tensor<float>({2, 2}, {1, 0, 0, 1}));
    model.addInitializer("b1", ie::Tensor<float>({2}, {0, 0}));

    return model;
}

}

TEST_CASE(model_loader_round_trips_graph_and_weights) {
    std::filesystem::path path = std::filesystem::temp_directory_path() / "engine_test_model.oien";

    ie::ModelLoader::save(buildSampleModel(), path.string());
    ie::Model reloaded = ie::ModelLoader::load(path.string());

    std::filesystem::remove(path);

    ASSERT_EQ(reloaded.graph().size(), 2u);
    ASSERT_EQ(reloaded.graph().inputs().size(), 1u);
    ASSERT_EQ(reloaded.graph().inputs()[0], std::string("input"));
    ASSERT_EQ(reloaded.graph().outputs()[0], std::string("output"));

    const auto& w1 = reloaded.initializers().at("w1");
    ASSERT_EQ(w1.dim(0), 2u);
    ASSERT_EQ(w1.dim(1), 2u);
    ASSERT_NEAR(w1[0], 1.0f, 1e-9);
    ASSERT_NEAR(w1[3], 1.0f, 1e-9);

    ie::InferenceEngine engine(std::move(reloaded));
    ie::Tensor<float> input({1, 2}, {3.0f, -4.0f});
    ie::Tensor<float> output = engine.infer(input);

    ASSERT_NEAR(output[0], 3.0f, 1e-5);
    ASSERT_NEAR(output[1], 0.0f, 1e-5);
}
