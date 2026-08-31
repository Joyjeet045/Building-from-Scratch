/*
 * allocator.cpp
 * Implementation of the aligned CPU allocator and the pooled arena allocator.
 */
#include "ie/core/allocator.h"

#include <new>

namespace ie {

namespace {

void* alignedAllocate(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    std::size_t rounded = (bytes + kAlignment - 1) / kAlignment * kAlignment;
    return ::operator new(rounded, std::align_val_t{kAlignment});
}

void alignedDeallocate(void* ptr, std::size_t bytes) {
    if (ptr == nullptr) return;
    std::size_t rounded = (bytes + kAlignment - 1) / kAlignment * kAlignment;
    ::operator delete(ptr, rounded, std::align_val_t{kAlignment});
}

}

CpuAllocator& CpuAllocator::instance() {
    static CpuAllocator allocator;
    return allocator;
}

void* CpuAllocator::allocate(std::size_t bytes) { return alignedAllocate(bytes); }

void CpuAllocator::deallocate(void* ptr, std::size_t bytes) { alignedDeallocate(ptr, bytes); }

const char* CpuAllocator::name() const { return "cpu"; }

PoolAllocator::PoolAllocator(std::size_t chunkBytes) : chunkBytes_(chunkBytes) {}

PoolAllocator::~PoolAllocator() {
    for (char* chunk : chunks_) {
        ::operator delete(chunk, std::align_val_t{kAlignment});
    }
}

std::size_t PoolAllocator::sizeClass(std::size_t bytes) {
    std::size_t rounded = (bytes + kAlignment - 1) / kAlignment * kAlignment;
    std::size_t cls = kAlignment;
    while (cls < rounded) cls <<= 1;
    return cls;
}

void PoolAllocator::addChunk(std::size_t minimumBytes) {
    std::size_t size = chunkBytes_ > minimumBytes ? chunkBytes_ : minimumBytes;
    char* chunk = static_cast<char*>(::operator new(size, std::align_val_t{kAlignment}));
    chunks_.push_back(chunk);
    cursor_ = chunk;
    remaining_ = size;
    stats_.arenaBytes += size;
}

void* PoolAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    std::size_t cls = sizeClass(bytes);

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = freeLists_.find(cls);
    if (it != freeLists_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        ++stats_.hits;
        stats_.liveBytes += cls;
        if (stats_.liveBytes > stats_.peakLiveBytes) stats_.peakLiveBytes = stats_.liveBytes;
        return ptr;
    }

    ++stats_.misses;
    if (remaining_ < cls) {
        addChunk(cls);
    }
    void* ptr = cursor_;
    cursor_ += cls;
    remaining_ -= cls;
    stats_.liveBytes += cls;
    if (stats_.liveBytes > stats_.peakLiveBytes) stats_.peakLiveBytes = stats_.liveBytes;
    return ptr;
}

void PoolAllocator::deallocate(void* ptr, std::size_t bytes) {
    if (ptr == nullptr || bytes == 0) return;
    std::size_t cls = sizeClass(bytes);

    std::lock_guard<std::mutex> lock(mutex_);
    freeLists_[cls].push_back(ptr);
    stats_.liveBytes -= cls;
}

const char* PoolAllocator::name() const { return "pool"; }

PoolStats PoolAllocator::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}
