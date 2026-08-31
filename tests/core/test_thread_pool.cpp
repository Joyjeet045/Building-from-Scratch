/*
 * test_thread_pool.cpp
 * Tests for the worker pool primitives that back both inter-op and intra-op
 * parallelism.
 */
#include <atomic>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "framework/test_framework.h"
#include "ie/core/thread_pool.h"

TEST_CASE(thread_pool_runs_every_task_exactly_once) {
    ie::ThreadPool pool(3);
    std::atomic<int> counter{0};

    std::vector<std::function<void()>> tasks;
    for (int i = 0; i < 32; ++i) {
        tasks.emplace_back([&counter] { counter.fetch_add(1); });
    }
    pool.runAll(tasks);

    ASSERT_EQ(counter.load(), 32);
}

TEST_CASE(thread_pool_parallel_for_covers_the_whole_range) {
    ie::ThreadPool pool(3);
    std::vector<int> marks(1000, 0);

    pool.parallelFor(marks.size(), [&marks](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) marks[i] = 1;
    });

    ASSERT_EQ(std::accumulate(marks.begin(), marks.end(), 0), 1000);
}

TEST_CASE(thread_pool_propagates_task_exceptions) {
    ie::ThreadPool pool(2);
    std::vector<std::function<void()>> tasks;
    tasks.emplace_back([] {});
    tasks.emplace_back([] { throw std::runtime_error("boom"); });
    tasks.emplace_back([] {});

    ASSERT_THROWS(pool.runAll(tasks));
}

TEST_CASE(thread_pool_nested_parallel_for_runs_inline_without_deadlock) {
    ie::ThreadPool pool(2);
    std::atomic<int> inner{0};

    std::vector<std::function<void()>> tasks;
    for (int i = 0; i < 4; ++i) {
        tasks.emplace_back([&pool, &inner] {
            pool.parallelFor(100, [&inner](std::size_t begin, std::size_t end) {
                inner.fetch_add(static_cast<int>(end - begin));
            });
        });
    }
    pool.runAll(tasks);

    ASSERT_EQ(inner.load(), 400);
}
