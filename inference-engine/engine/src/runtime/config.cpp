/*
 * config.cpp
 * Parsing and validation of the flat key/value configuration file.
 */
#include "ie/runtime/config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ie {

namespace {

std::string trim(const std::string& value) {
    std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool parseBool(const std::string& value, const std::string& key) {
    if (value == "true" || value == "1" || value == "yes") return true;
    if (value == "false" || value == "0" || value == "no") return false;
    throw std::invalid_argument("Config key '" + key + "' expects a boolean, got '" + value + "'");
}

std::size_t parseSize(const std::string& value, const std::string& key) {
    try {
        long long parsed = std::stoll(value);
        if (parsed < 0) throw std::out_of_range("negative");
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument("Config key '" + key + "' expects a non-negative integer, got '" +
                                     value + "'");
    }
}

}

SessionConfig SessionConfig::fromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Failed to open config file: " + path);

    SessionConfig config;
    std::string line;
    std::size_t lineNumber = 0;

    while (std::getline(file, line)) {
        ++lineNumber;
        std::size_t comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;

        std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            throw std::invalid_argument("Malformed config at line " + std::to_string(lineNumber) + ": " +
                                         line);
        }

        std::string key = trim(line.substr(0, colon));
        std::string value = trim(line.substr(colon + 1));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }

        if (key == "name") config.name = value;
        else if (key == "model_path") config.modelPath = value;
        else if (key == "execution_provider") config.executionProvider = value;
        else if (key == "gemm_strategy") config.gemmStrategy = value;
        else if (key == "max_batch_size") config.maxBatchSize = parseSize(value, key);
        else if (key == "intra_op_threads") config.intraOpThreads = parseSize(value, key);
        else if (key == "inter_op_parallel") config.interOpParallel = parseBool(value, key);
        else if (key == "graph_optimization") config.graphOptimization = parseBool(value, key);
        else if (key == "memory_pool") config.memoryPool = parseBool(value, key);
        else if (key == "profiling") config.profiling = parseBool(value, key);
        else if (key == "max_queue_delay_microseconds") config.maxQueueDelayMicroseconds = parseSize(value, key);
        else if (key == "http_address") config.httpAddress = value;
        else if (key == "http_port") config.httpPort = static_cast<int>(parseSize(value, key));
        else throw std::invalid_argument("Unknown config key '" + key + "' at line " + std::to_string(lineNumber));
    }

    config.validate();
    return config;
}

void SessionConfig::validate() const {
    if (modelPath.empty()) {
        throw std::invalid_argument("Config must set model_path");
    }
    if (maxBatchSize == 0 || maxBatchSize > 4096) {
        throw std::invalid_argument("max_batch_size must be between 1 and 4096");
    }
    if (intraOpThreads == 0 || intraOpThreads > 256) {
        throw std::invalid_argument("intra_op_threads must be between 1 and 256");
    }
    if (httpPort <= 0 || httpPort > 65535) {
        throw std::invalid_argument("http_port must be between 1 and 65535");
    }
    gemmStrategyFromString(gemmStrategy);
    if (executionProvider != "cpu") {
        throw std::invalid_argument("Unsupported execution_provider: " + executionProvider);
    }
}

std::string SessionConfig::toString() const {
    std::ostringstream oss;
    oss << "name: " << name << "\n"
        << "model_path: " << modelPath << "\n"
        << "execution_provider: " << executionProvider << "\n"
        << "gemm_strategy: " << gemmStrategy << "\n"
        << "max_batch_size: " << maxBatchSize << "\n"
        << "intra_op_threads: " << intraOpThreads << "\n"
        << "inter_op_parallel: " << (interOpParallel ? "true" : "false") << "\n"
        << "graph_optimization: " << (graphOptimization ? "true" : "false") << "\n"
        << "memory_pool: " << (memoryPool ? "true" : "false") << "\n"
        << "profiling: " << (profiling ? "true" : "false") << "\n"
        << "max_queue_delay_microseconds: " << maxQueueDelayMicroseconds << "\n"
        << "http_address: " << httpAddress << "\n"
        << "http_port: " << httpPort << "\n";
    return oss.str();
}

CpuProviderOptions SessionConfig::cpuOptions() const {
    CpuProviderOptions options;
    options.gemmStrategy = gemmStrategyFromString(gemmStrategy);
    options.intraOpThreads = intraOpThreads;
    options.useMemoryPool = memoryPool;
    return options;
}

OptimizerOptions SessionConfig::optimizerOptions() const {
    OptimizerOptions options;
    options.fuseActivations = graphOptimization;
    options.foldConstants = graphOptimization;
    options.eliminateDeadNodes = graphOptimization;
    return options;
}

}
