/*
 * profiler.h
 * Per-node timing collection standing in for the flame charts used in the
 * profiling write-up: it answers the same question, namely what share of the
 * inference loop is spent on real operator work versus everything else.
 */
#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ie {

struct NodeTiming {
    std::string node;
    std::string opType;
    uint64_t calls = 0;
    double totalMs = 0.0;
};

class Profiler {
public:
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    void record(const std::string& node, const std::string& opType, double elapsedMs);
    void recordTotal(double elapsedMs);
    void reset();

    std::vector<NodeTiming> timings() const;
    double totalMs() const { return totalMs_; }
    uint64_t runs() const { return runs_; }
    std::string report() const;

private:
    bool enabled_ = false;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, NodeTiming> timings_;
    double totalMs_ = 0.0;
    uint64_t runs_ = 0;
};

class ScopedTimer {
public:
    ScopedTimer() : start_(std::chrono::steady_clock::now()) {}

    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

}
