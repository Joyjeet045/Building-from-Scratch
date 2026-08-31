/*
 * thread_pool.h
 * Fixed-size worker pool backing two kinds of parallelism: inter-op, where
 * independent graph branches run concurrently, and intra-op, where a single
 * operator splits its output rows across workers. Nested parallelism is
 * deliberately not supported - parallelFor called from a worker runs inline -
 * so a blocked worker can never wait on a task only another worker could run.
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ie {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    static std::size_t hardwareThreads();

    std::size_t size() const { return workers_.size(); }

    bool insidePool() const;

    void runAll(const std::vector<std::function<void()>>& tasks);

    void parallelFor(std::size_t count, const std::function<void(std::size_t, std::size_t)>& body,
                     std::size_t minChunk = 1);

private:
    void workerLoop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = false;
};

}
