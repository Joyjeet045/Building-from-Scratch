/*
 * main.cpp
 * Command-line front end.
 *   run     <model> <image>          classify a single 28x28 uint8 image
 *   eval    <model> <dataset> [opts] accuracy and latency over a whole dataset
 *   bench   <model> <dataset> [opts] throughput sweep across gemm strategies,
 *                                    batch sizes, and scheduling modes
 *   inspect <model> [opts]           optimization report and parallel schedule
 *   serve   <config>                 HTTP server with dynamic batching
 * Options: --config <file> --batch N --threads N --gemm <strategy>
 *          --parallel --no-optimize --profile
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "ie/core/tensor.h"
#include "ie/model/model_loader.h"
#include "ie/runtime/config.h"
#include "ie/runtime/inference_engine.h"
#include "ie/serving/dynamic_batcher.h"
#include "ie/serving/http_server.h"

namespace {

constexpr std::size_t kImagePixels = 28 * 28;

struct Options {
    ie::SessionConfig config;
    std::size_t batchSize = 1;
};

std::vector<float> normalize(const uint8_t* pixels, std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(pixels[i]) / 255.0f;
    }
    return values;
}

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

struct Dataset {
    std::vector<uint8_t> pixels;
    std::vector<uint8_t> labels;
    std::size_t count = 0;
};

Dataset loadDataset(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open dataset file: " + path);

    uint32_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file || count == 0 || count > 10000000u) {
        throw std::runtime_error("Dataset file declares an invalid sample count");
    }

    Dataset dataset;
    dataset.count = count;
    dataset.pixels.resize(static_cast<std::size_t>(count) * kImagePixels);
    dataset.labels.resize(count);

    std::vector<uint8_t> record(kImagePixels + 1);
    for (uint32_t i = 0; i < count; ++i) {
        file.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(record.size()));
        if (!file) throw std::runtime_error("Dataset truncated at sample " + std::to_string(i));
        std::copy(record.begin(), record.begin() + kImagePixels,
                   dataset.pixels.begin() + static_cast<std::size_t>(i) * kImagePixels);
        dataset.labels[i] = record[kImagePixels];
    }
    return dataset;
}

std::size_t argmax(const float* values, std::size_t count) {
    return static_cast<std::size_t>(std::distance(values, std::max_element(values, values + count)));
}

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    return sorted[static_cast<std::size_t>(p * (sorted.size() - 1))];
}

Options parseOptions(int argc, char** argv, int firstOption, const std::string& modelPath) {
    Options options;
    options.config.modelPath = modelPath;

    for (int i = firstOption; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("Missing value for ") + what);
            return argv[++i];
        };

        if (arg == "--config") {
            std::string keepModel = options.config.modelPath;
            options.config = ie::SessionConfig::fromFile(next("--config"));
            if (!keepModel.empty()) options.config.modelPath = keepModel;
        } else if (arg == "--batch") {
            options.batchSize = static_cast<std::size_t>(std::stoul(next("--batch")));
        } else if (arg == "--threads") {
            options.config.intraOpThreads = static_cast<std::size_t>(std::stoul(next("--threads")));
        } else if (arg == "--gemm") {
            options.config.gemmStrategy = next("--gemm");
        } else if (arg == "--parallel") {
            options.config.interOpParallel = true;
        } else if (arg == "--no-optimize") {
            options.config.graphOptimization = false;
        } else if (arg == "--profile") {
            options.config.profiling = true;
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (options.batchSize == 0) options.batchSize = 1;
    if (options.batchSize > options.config.maxBatchSize) {
        options.config.maxBatchSize = options.batchSize;
    }
    options.config.validate();
    return options;
}

struct EvalResult {
    std::size_t correct = 0;
    std::size_t samples = 0;
    double meanMs = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double throughput = 0.0;
};

EvalResult evaluate(ie::InferenceEngine& engine, const Dataset& dataset, std::size_t batchSize) {
    EvalResult result;
    result.samples = dataset.count;

    std::vector<double> latencies;
    latencies.reserve(dataset.count / batchSize + 1);

    ie::Tensor<float> input({static_cast<uint64_t>(batchSize), kImagePixels});
    double totalMs = 0.0;

    for (std::size_t offset = 0; offset < dataset.count; offset += batchSize) {
        std::size_t current = std::min(batchSize, dataset.count - offset);
        if (current != input.dim(0)) {
            input = ie::Tensor<float>({static_cast<uint64_t>(current), kImagePixels});
        }

        float* dst = input.data();
        for (std::size_t i = 0; i < current; ++i) {
            const uint8_t* src = dataset.pixels.data() + (offset + i) * kImagePixels;
            for (std::size_t p = 0; p < kImagePixels; ++p) {
                dst[i * kImagePixels + p] = static_cast<float>(src[p]) / 255.0f;
            }
        }

        auto start = std::chrono::steady_clock::now();
        const ie::Tensor<float>& output = engine.inferBatch(input);
        auto end = std::chrono::steady_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        totalMs += elapsed;
        latencies.push_back(elapsed);

        uint64_t classes = output.dim(1);
        const float* logits = output.data();
        for (std::size_t i = 0; i < current; ++i) {
            if (argmax(logits + i * classes, classes) == dataset.labels[offset + i]) {
                ++result.correct;
            }
        }
    }

    std::sort(latencies.begin(), latencies.end());
    result.meanMs = totalMs / static_cast<double>(latencies.size());
    result.p50 = percentile(latencies, 0.50);
    result.p95 = percentile(latencies, 0.95);
    result.p99 = percentile(latencies, 0.99);
    result.throughput = 1000.0 * static_cast<double>(dataset.count) / totalMs;
    return result;
}

int runSingle(const std::string& imagePath, const Options& options) {
    std::vector<uint8_t> raw = readFile(imagePath);
    if (raw.size() != kImagePixels) {
        throw std::runtime_error("Expected a 784-byte (28x28) raw uint8 image, got " +
                                  std::to_string(raw.size()) + " bytes");
    }

    ie::InferenceEngine engine(options.config);
    ie::Tensor<float> input({1, 28, 28}, normalize(raw.data(), raw.size()));

    engine.infer(input);

    auto start = std::chrono::steady_clock::now();
    ie::Tensor<float> output = engine.infer(input);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Out: " << output.toString() << "\n";
    std::cout << "Predicted digit: " << argmax(output.data(), output.size()) << "\n";
    std::cout << "Inference latency: " << std::chrono::duration<double, std::milli>(end - start).count()
               << " ms\n";

    if (options.config.profiling) {
        std::cout << "\nPer-node profile:\n" << engine.session().profiler().report();
    }
    return 0;
}

int runEval(const std::string& datasetPath, const Options& options) {
    Dataset dataset = loadDataset(datasetPath);
    ie::InferenceEngine engine(options.config);

    std::cout << "Graph optimization: " << engine.session().optimizationReport().toString() << "\n";
    std::cout << "Schedule: " << engine.session().levelCount() << " levels, "
               << engine.session().parallelLevelCount() << " parallelizable\n";
    std::cout << "Provider: " << engine.session().provider().name()
               << " | gemm: " << options.config.gemmStrategy
               << " | threads: " << options.config.intraOpThreads
               << " | inter-op: " << (options.config.interOpParallel ? "on" : "off")
               << " | batch: " << options.batchSize << "\n\n";

    EvalResult result = evaluate(engine, dataset, options.batchSize);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Samples: " << result.samples << "\n";
    std::cout << "Correct: " << result.correct << "\n";
    std::cout << "Accuracy: " << (100.0 * static_cast<double>(result.correct) / result.samples) << "%\n";
    std::cout << "Latency mean: " << result.meanMs << " ms per batch\n";
    std::cout << "Latency p50: " << result.p50 << " ms\n";
    std::cout << "Latency p95: " << result.p95 << " ms\n";
    std::cout << "Latency p99: " << result.p99 << " ms\n";
    std::cout << "Throughput: " << std::setprecision(1) << result.throughput << " inferences/sec\n";

    if (options.config.profiling) {
        std::cout << "\nPer-node profile:\n" << engine.session().profiler().report();
    }
    return 0;
}

int runBench(const std::string& datasetPath, const Options& options) {
    Dataset full = loadDataset(datasetPath);

    Dataset sample;
    sample.count = std::min<std::size_t>(full.count, 2000);
    sample.pixels.assign(full.pixels.begin(), full.pixels.begin() + sample.count * kImagePixels);
    sample.labels.assign(full.labels.begin(), full.labels.begin() + sample.count);

    std::cout << "Benchmarking on " << sample.count << " samples.\n\n";

    std::cout << "GEMM strategy sweep (batch 1, single thread)\n";
    std::cout << "  " << std::left << std::setw(12) << "strategy" << std::right << std::setw(14)
               << "inf/sec" << std::setw(15) << "us/inference" << std::setw(12) << "accuracy" << "\n";
    for (const char* strategy : {"naive", "reordered", "tiled", "packed", "auto"}) {
        ie::SessionConfig config = options.config;
        config.gemmStrategy = strategy;
        config.intraOpThreads = 1;
        config.interOpParallel = false;
        ie::InferenceEngine engine(config);
        EvalResult result = evaluate(engine, sample, 1);
        std::cout << "  " << std::left << std::setw(12) << strategy << std::right << std::fixed
                   << std::setprecision(1) << std::setw(14) << result.throughput << std::setw(15)
                   << (1e6 / result.throughput) << std::setw(11)
                   << (100.0 * static_cast<double>(result.correct) / result.samples) << "%\n";
    }

    std::cout << "\nBatch size sweep (auto gemm, single thread)\n";
    std::cout << "  " << std::left << std::setw(12) << "batch" << std::right << std::setw(14) << "inf/sec"
               << std::setw(15) << "us/inference" << std::setw(12) << "speedup" << "\n";
    double baseline = 0.0;
    for (std::size_t batch : {1, 2, 4, 8, 16, 32, 64, 128}) {
        ie::SessionConfig config = options.config;
        config.maxBatchSize = std::max<std::size_t>(config.maxBatchSize, batch);
        config.intraOpThreads = 1;
        config.interOpParallel = false;
        ie::InferenceEngine engine(config);
        EvalResult result = evaluate(engine, sample, batch);
        if (baseline == 0.0) baseline = result.throughput;
        std::cout << "  " << std::left << std::setw(12) << batch << std::right << std::fixed
                   << std::setprecision(1) << std::setw(14) << result.throughput << std::setw(15)
                   << (1e6 / result.throughput) << std::setw(11) << std::setprecision(2)
                   << (result.throughput / baseline) << "x\n";
    }

    std::size_t hw = ie::ThreadPool::hardwareThreads();
    std::cout << "\nScheduling sweep (batch 32, " << hw << " hardware threads available)\n";
    std::cout << "  " << std::left << std::setw(28) << "mode" << std::right << std::setw(14) << "inf/sec"
               << std::setw(12) << "speedup" << "\n";
    struct Mode {
        const char* label;
        std::size_t threads;
        bool interOp;
    };
    std::size_t workers = std::min<std::size_t>(hw, 4);
    std::vector<Mode> modes{{"sequential, 1 thread", 1, false},
                             {"inter-op branch parallel", workers, true},
                             {"intra-op gemm row split", workers, false}};
    double schedBaseline = 0.0;
    for (const Mode& mode : modes) {
        ie::SessionConfig config = options.config;
        config.maxBatchSize = std::max<std::size_t>(config.maxBatchSize, 32);
        config.intraOpThreads = mode.threads;
        config.interOpParallel = mode.interOp;
        ie::InferenceEngine engine(config);
        EvalResult result = evaluate(engine, sample, 32);
        if (schedBaseline == 0.0) schedBaseline = result.throughput;
        std::cout << "  " << std::left << std::setw(28) << mode.label << std::right << std::fixed
                   << std::setprecision(1) << std::setw(14) << result.throughput << std::setw(11)
                   << std::setprecision(2) << (result.throughput / schedBaseline) << "x\n";
    }

    std::cout << "\nGraph optimization impact (batch 1)\n";
    for (bool optimize : {false, true}) {
        ie::SessionConfig config = options.config;
        config.graphOptimization = optimize;
        config.intraOpThreads = 1;
        config.interOpParallel = false;
        ie::InferenceEngine engine(config);
        EvalResult result = evaluate(engine, sample, 1);
        std::cout << "  " << std::left << std::setw(14) << (optimize ? "optimized" : "unoptimized")
                   << std::right << std::fixed << std::setprecision(1) << std::setw(12)
                   << result.throughput << "   " << engine.session().optimizationReport().toString()
                   << "\n";
    }

    return 0;
}

int runInspect(const Options& options) {
    ie::InferenceEngine engine(options.config);
    const ie::InferenceSession& session = engine.session();

    std::cout << "Config:\n" << options.config.toString() << "\n";
    std::cout << "Graph optimization: " << session.optimizationReport().toString() << "\n\n";
    std::cout << "Execution schedule (" << session.levelCount() << " levels, "
               << session.parallelLevelCount() << " parallelizable):\n";
    std::cout << session.describeSchedule();
    return 0;
}

int runServe(const std::string& configPath) {
    ie::SessionConfig config = ie::SessionConfig::fromFile(configPath);
    ie::InferenceEngine engine(config);

    std::cout << "Loaded model '" << config.name << "' from " << config.modelPath << "\n";
    std::cout << "Graph optimization: " << engine.session().optimizationReport().toString() << "\n";
    std::cout << "Dynamic batching: max_batch_size=" << config.maxBatchSize
               << " max_queue_delay_us=" << config.maxQueueDelayMicroseconds << "\n";

    ie::DynamicBatcher batcher(engine, config.maxBatchSize, config.maxQueueDelayMicroseconds, kImagePixels);
    ie::HttpServer server(batcher, config.httpAddress, config.httpPort, kImagePixels);
    server.run();
    return 0;
}

void printUsage(const char* program) {
    std::cerr << "Usage:\n"
               << "  " << program << " run     <model.oien> <image.ubyte> [options]\n"
               << "  " << program << " eval    <model.oien> <dataset.bin> [options]\n"
               << "  " << program << " bench   <model.oien> <dataset.bin> [options]\n"
               << "  " << program << " inspect <model.oien>               [options]\n"
               << "  " << program << " serve   <config.yaml>\n\n"
               << "Options:\n"
               << "  --config <file>   load a Triton-style config file\n"
               << "  --batch <n>       batch size for eval\n"
               << "  --threads <n>     worker threads (intra-op and inter-op)\n"
               << "  --gemm <s>        naive | reordered | tiled | packed | auto\n"
               << "  --parallel        run independent graph branches concurrently\n"
               << "  --no-optimize     disable graph optimization passes\n"
               << "  --profile         collect and print per-node timings\n";
}

}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }

        std::string command = argv[1];

        if (command == "serve") {
            return runServe(argv[2]);
        }
        if (command == "inspect") {
            return runInspect(parseOptions(argc, argv, 3, argv[2]));
        }
        if (command == "run" || command == "eval" || command == "bench") {
            if (argc < 4) {
                printUsage(argv[0]);
                return 1;
            }
            Options options = parseOptions(argc, argv, 4, argv[2]);
            if (command == "run") return runSingle(argv[3], options);
            if (command == "eval") return runEval(argv[3], options);
            return runBench(argv[3], options);
        }
        if (argc == 3) {
            return runSingle(argv[2], parseOptions(argc, argv, 3, argv[1]));
        }

        printUsage(argv[0]);
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
