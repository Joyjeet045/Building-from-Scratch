/*
 * thread_pool.cpp
 * Implementation of the worker pool. Both runAll and parallelFor keep the
 * calling thread busy on one share of the work while the pool handles the
 * rest, and both propagate the first exception thrown by any task.
 */
#include "ie/core/thread_pool.h"

#include <atomic>
#include <exception>
#include <stdexcept>

namespace ie {

namespace {

thread_local const ThreadPool* tCurrentPool = nullptr;

struct TaskGroup {
    std::mutex mutex;
    std::condition_variable done;
    std::size_t remaining = 0;
    std::exception_ptr error;
};

}

ThreadPool::ThreadPool(std::size_t threads) {
    workers_.reserve(threads);
    for (std::size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

std::size_t ThreadPool::hardwareThreads() {
    unsigned int detected = std::thread::hardware_concurrency();
    return detected == 0 ? 1 : static_cast<std::size_t>(detected);
}

bool ThreadPool::insidePool() const { return tCurrentPool == this; }

void ThreadPool::workerLoop() {
    tCurrentPool = this;
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) break;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
    tCurrentPool = nullptr;
}

void ThreadPool::runAll(const std::vector<std::function<void()>>& tasks) {
    if (tasks.empty()) return;
    if (tasks.size() == 1 || workers_.empty() || insidePool()) {
        for (const auto& task : tasks) task();
        return;
    }

    TaskGroup group;
    group.remaining = tasks.size() - 1;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 1; i < tasks.size(); ++i) {
            const auto& task = tasks[i];
            tasks_.emplace([&group, &task] {
                try {
                    task();
                } catch (...) {
                    std::lock_guard<std::mutex> guard(group.mutex);
                    if (!group.error) group.error = std::current_exception();
                }
                std::lock_guard<std::mutex> guard(group.mutex);
                if (--group.remaining == 0) group.done.notify_all();
            });
        }
    }
    condition_.notify_all();

    std::exception_ptr callerError;
    try {
        tasks[0]();
    } catch (...) {
        callerError = std::current_exception();
    }

    {
        std::unique_lock<std::mutex> lock(group.mutex);
        group.done.wait(lock, [&group] { return group.remaining == 0; });
    }

    if (callerError) std::rethrow_exception(callerError);
    if (group.error) std::rethrow_exception(group.error);
}

void ThreadPool::parallelFor(std::size_t count, const std::function<void(std::size_t, std::size_t)>& body,
                              std::size_t minChunk) {
    if (count == 0) return;
    if (minChunk == 0) minChunk = 1;

    std::size_t maxShards = (count + minChunk - 1) / minChunk;
    std::size_t shards = workers_.size() + 1;
    if (shards > maxShards) shards = maxShards;

    if (shards <= 1 || workers_.empty() || insidePool()) {
        body(0, count);
        return;
    }

    std::size_t chunk = (count + shards - 1) / shards;
    std::vector<std::function<void()>> tasks;
    tasks.reserve(shards);
    for (std::size_t s = 0; s < shards; ++s) {
        std::size_t begin = s * chunk;
        if (begin >= count) break;
        std::size_t end = begin + chunk;
        if (end > count) end = count;
        tasks.emplace_back([&body, begin, end] { body(begin, end); });
    }
    runAll(tasks);
}

}
