/*
 * allocator.h
 * Pluggable memory allocation for Tensor buffers. CpuAllocator is a thin
 * 64-byte-aligned wrapper over global new/delete; PoolAllocator pre-reserves
 * arena chunks and recycles freed blocks through size-class free lists so the
 * inference loop performs no real allocations after warm-up.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ie {

constexpr std::size_t kAlignment = 64;

class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* ptr, std::size_t bytes) = 0;
    virtual const char* name() const = 0;
};

class CpuAllocator final : public Allocator {
public:
    static CpuAllocator& instance();

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr, std::size_t bytes) override;
    const char* name() const override;
};

struct PoolStats {
    std::size_t arenaBytes = 0;
    std::size_t liveBytes = 0;
    std::size_t peakLiveBytes = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
};

class PoolAllocator final : public Allocator {
public:
    explicit PoolAllocator(std::size_t chunkBytes = 4 * 1024 * 1024);
    ~PoolAllocator() override;

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr, std::size_t bytes) override;
    const char* name() const override;

    PoolStats stats() const;

private:
    static std::size_t sizeClass(std::size_t bytes);
    void addChunk(std::size_t minimumBytes);

    mutable std::mutex mutex_;
    std::size_t chunkBytes_;
    std::vector<char*> chunks_;
    char* cursor_ = nullptr;
    std::size_t remaining_ = 0;
    std::unordered_map<std::size_t, std::vector<void*>> freeLists_;
    PoolStats stats_;
};

}
