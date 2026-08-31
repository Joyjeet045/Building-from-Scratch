/*
 * test_config.cpp
 * Tests for parsing the Triton-style config file and for the validation that
 * rejects out-of-range settings before a session is ever built.
 */
#include <filesystem>
#include <fstream>

#include "framework/test_framework.h"
#include "ie/runtime/config.h"

TEST_CASE(config_parses_and_round_trips) {
    std::filesystem::path path = std::filesystem::temp_directory_path() / "engine_test_config.yaml";
    {
        std::ofstream out(path);
        out << "# sample config\n"
             << "name: mnist\n"
             << "model_path: \"models/mnist_ffn.oien\"\n"
             << "execution_provider: cpu\n"
             << "gemm_strategy: packed\n"
             << "max_batch_size: 64\n"
             << "intra_op_threads: 4\n"
             << "inter_op_parallel: true\n"
             << "graph_optimization: false\n"
             << "http_port: 9000\n";
    }

    ie::SessionConfig config = ie::SessionConfig::fromFile(path.string());
    std::filesystem::remove(path);

    ASSERT_EQ(config.name, std::string("mnist"));
    ASSERT_EQ(config.modelPath, std::string("models/mnist_ffn.oien"));
    ASSERT_EQ(config.gemmStrategy, std::string("packed"));
    ASSERT_EQ(config.maxBatchSize, 64u);
    ASSERT_EQ(config.intraOpThreads, 4u);
    ASSERT_TRUE(config.interOpParallel);
    ASSERT_TRUE(!config.graphOptimization);
    ASSERT_EQ(config.httpPort, 9000);
}

TEST_CASE(config_rejects_unknown_keys) {
    std::filesystem::path path = std::filesystem::temp_directory_path() / "engine_bad_config.yaml";
    {
        std::ofstream out(path);
        out << "model_path: model.oien\n" << "not_a_real_key: 1\n";
    }

    bool threw = false;
    try {
        ie::SessionConfig::fromFile(path.string());
    } catch (const std::exception&) {
        threw = true;
    }
    std::filesystem::remove(path);
    ASSERT_TRUE(threw);
}

TEST_CASE(config_rejects_out_of_range_values) {
    ie::SessionConfig config;
    config.modelPath = "model.oien";
    config.httpPort = 0;
    ASSERT_THROWS(config.validate());

    ie::SessionConfig missingModel;
    ASSERT_THROWS(missingModel.validate());

    ie::SessionConfig badStrategy;
    badStrategy.modelPath = "model.oien";
    badStrategy.gemmStrategy = "nonexistent";
    ASSERT_THROWS(badStrategy.validate());

    ie::SessionConfig badProvider;
    badProvider.modelPath = "model.oien";
    badProvider.executionProvider = "quantum";
    ASSERT_THROWS(badProvider.validate());
}

TEST_CASE(config_missing_file_reports_an_error) {
    ASSERT_THROWS(ie::SessionConfig::fromFile("definitely/not/here.yaml"));
}
