/*
 * profiler.cpp
 * Aggregation and formatting of collected per-node timings.
 */
#include "ie/runtime/profiler.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace ie {

void Profiler::record(const std::string& node, const std::string& opType, double elapsedMs) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    NodeTiming& timing = timings_[node];
    if (timing.calls == 0) {
        timing.node = node;
        timing.opType = opType;
    }
    ++timing.calls;
    timing.totalMs += elapsedMs;
}

void Profiler::recordTotal(double elapsedMs) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    totalMs_ += elapsedMs;
    ++runs_;
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    timings_.clear();
    totalMs_ = 0.0;
    runs_ = 0;
}

std::vector<NodeTiming> Profiler::timings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<NodeTiming> result;
    result.reserve(timings_.size());
    for (const auto& [name, timing] : timings_) result.push_back(timing);
    std::sort(result.begin(), result.end(),
               [](const NodeTiming& a, const NodeTiming& b) { return a.totalMs > b.totalMs; });
    return result;
}

std::string Profiler::report() const {
    std::vector<NodeTiming> sorted = timings();
    double nodeTotal = 0.0;
    for (const auto& timing : sorted) nodeTotal += timing.totalMs;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "  " << std::left << std::setw(14) << "node" << std::setw(11) << "op" << std::right
        << std::setw(9) << "calls" << std::setw(12) << "total ms" << std::setw(12) << "us/call"
        << std::setw(9) << "share\n";
    for (const auto& timing : sorted) {
        double share = nodeTotal > 0.0 ? 100.0 * timing.totalMs / nodeTotal : 0.0;
        oss << "  " << std::left << std::setw(14) << timing.node << std::setw(11) << timing.opType
            << std::right << std::setw(9) << timing.calls << std::setw(12) << timing.totalMs
            << std::setw(12) << (timing.totalMs * 1000.0 / static_cast<double>(timing.calls))
            << std::setw(8) << share << "%\n";
    }
    if (runs_ > 0) {
        double overhead = totalMs_ > 0.0 ? 100.0 * (totalMs_ - nodeTotal) / totalMs_ : 0.0;
        oss << "  operator time: " << nodeTotal << " ms of " << totalMs_ << " ms wall ("
            << (100.0 - overhead) << "% useful work, " << overhead << "% scheduling overhead)\n";
    }
    return oss.str();
}

}
