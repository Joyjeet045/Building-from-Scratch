/*
 * test_allocator.cpp
 * Tests for the pooled arena allocator and the alignment guarantee Tensor
 * relies on for SIMD-friendly loads.
 */
#include <cstdint>

#include "framework/test_framework.h"
#include "ie/core/allocator.h"
#include "ie/core/tensor.h"

TEST_CASE(pool_allocator_recycles_freed_blocks) {
    ie::PoolAllocator pool(1024 * 1024);

    void* first = pool.allocate(4096);
    pool.deallocate(first, 4096);
    void* second = pool.allocate(4096);

    ASSERT_TRUE(first == second);
    ASSERT_EQ(pool.stats().hits, 1u);
    pool.deallocate(second, 4096);
}

TEST_CASE(pool_allocator_tracks_peak_usage) {
    ie::PoolAllocator pool(1024 * 1024);
    void* a = pool.allocate(1024);
    void* b = pool.allocate(2048);
    ASSERT_TRUE(pool.stats().peakLiveBytes >= 3072u);
    pool.deallocate(a, 1024);
    pool.deallocate(b, 2048);
    ASSERT_EQ(pool.stats().liveBytes, 0u);
}

TEST_CASE(pool_allocator_grows_when_a_chunk_is_exhausted) {
    ie::PoolAllocator pool(4096);
    std::vector<void*> blocks;
    for (int i = 0; i < 16; ++i) blocks.push_back(pool.allocate(1024));

    ASSERT_TRUE(pool.stats().arenaBytes > 4096u);
    for (void* block : blocks) pool.deallocate(block, 1024);
}

TEST_CASE(tensor_buffers_are_cache_line_aligned) {
    ie::Tensor<float> tensor({37});
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(tensor.data()) % ie::kAlignment, 0u);

    ie::PoolAllocator pool;
    ie::Tensor<float> pooled({91}, &pool);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(pooled.data()) % ie::kAlignment, 0u);
    ASSERT_TRUE(pooled.allocator() == &pool);
}
