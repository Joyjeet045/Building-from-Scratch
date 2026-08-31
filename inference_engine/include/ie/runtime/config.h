/*
 * config.h
 * Model configuration loaded from a Triton-style config file. Only the flat
 * "key: value" subset of YAML is supported, which is all the engine needs and
 * keeps the project free of a YAML dependency. Comments start with '#'.
 */
#pragma once

#include <cstddef>
#include <string>

#include "ie/graph/graph_optimizer.h"
#include "ie/providers/cpu_execution_provider.h"

namespace ie {

struct SessionConfig {
    std::string name = "model";
    std::string modelPath;
    std::string executionProvider = "cpu";
    std::string gemmStrategy = "auto";

    std::size_t maxBatchSize = 128;
    std::size_t intraOpThreads = 1;
    bool interOpParallel = false;
    bool graphOptimization = true;
    bool memoryPool = true;
    bool profiling = false;

    std::size_t maxQueueDelayMicroseconds = 2000;
    std::string httpAddress = "127.0.0.1";
    int httpPort = 8080;

    static SessionConfig fromFile(const std::string& path);
    void validate() const;
    std::string toString() const;

    CpuProviderOptions cpuOptions() const;
    OptimizerOptions optimizerOptions() const;
};

}
