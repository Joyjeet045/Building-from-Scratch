/*
 * tensor.h
 * Defines Tensor<T>: a row-major buffer plus shape. The buffer comes from a
 * pluggable Allocator (defaulting to a 64-byte-aligned CPU allocator) so that
 * an execution provider can hand tensors memory from a pool and so the data
 * pointer is always aligned for SIMD loads.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ie/core/allocator.h"

namespace ie {

template <typename T>
class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<uint64_t> shape, Allocator* allocator = nullptr)
        : shape_(std::move(shape)), allocator_(allocator ? allocator : &CpuAllocator::instance()) {
        size_ = computeSize(shape_);
        allocateBuffer();
        std::fill(data_, data_ + size_, T{});
    }

    Tensor(std::vector<uint64_t> shape, std::initializer_list<T> values, Allocator* allocator = nullptr)
        : Tensor(std::move(shape), std::vector<T>(values), allocator) {}

    Tensor(std::vector<uint64_t> shape, const std::vector<T>& values, Allocator* allocator = nullptr)
        : shape_(std::move(shape)), allocator_(allocator ? allocator : &CpuAllocator::instance()) {
        size_ = computeSize(shape_);
        if (values.size() != size_) {
            throw std::invalid_argument("Tensor data size does not match shape");
        }
        allocateBuffer();
        std::copy(values.begin(), values.end(), data_);
    }

    Tensor(const Tensor& other)
        : shape_(other.shape_), size_(other.size_), allocator_(other.allocator_) {
        allocateBuffer();
        std::copy(other.data_, other.data_ + size_, data_);
    }

    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)),
          data_(other.data_),
          size_(other.size_),
          allocator_(other.allocator_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.shape_.clear();
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;
        releaseBuffer();
        shape_ = other.shape_;
        size_ = other.size_;
        allocator_ = other.allocator_;
        allocateBuffer();
        std::copy(other.data_, other.data_ + size_, data_);
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        releaseBuffer();
        shape_ = std::move(other.shape_);
        data_ = other.data_;
        size_ = other.size_;
        allocator_ = other.allocator_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.shape_.clear();
        return *this;
    }

    ~Tensor() { releaseBuffer(); }

    static Tensor<T> zeros(std::vector<uint64_t> shape, Allocator* allocator = nullptr) {
        return Tensor<T>(std::move(shape), allocator);
    }

    const std::vector<uint64_t>& shape() const { return shape_; }

    uint64_t size() const { return size_; }

    std::size_t rank() const { return shape_.size(); }

    uint64_t dim(std::size_t axis) const { return shape_.at(axis); }

    Allocator* allocator() const { return allocator_; }

    T* data() { return data_; }

    const T* data() const { return data_; }

    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }

    T& operator[](uint64_t idx) { return data_[idx]; }

    const T& operator[](uint64_t idx) const { return data_[idx]; }

    T& at(uint64_t idx) {
        if (idx >= size_) throw std::out_of_range("Tensor index out of range");
        return data_[idx];
    }

    const T& at(uint64_t idx) const {
        if (idx >= size_) throw std::out_of_range("Tensor index out of range");
        return data_[idx];
    }

    std::vector<T> toVector() const { return std::vector<T>(data_, data_ + size_); }

    void reshape(std::vector<uint64_t> newShape) {
        if (computeSize(newShape) != size_) {
            throw std::invalid_argument("Cannot reshape tensor: element count mismatch");
        }
        shape_ = std::move(newShape);
    }

    bool operator==(const Tensor<T>& other) const {
        if (shape_ != other.shape_ || size_ != other.size_) return false;
        return std::equal(data_, data_ + size_, other.data_);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Tensor((";
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            oss << shape_[i];
            if (i + 1 < shape_.size()) oss << ", ";
        }
        oss << ")";
        if (shape_.size() == 2) {
            uint64_t rows = shape_[0];
            uint64_t cols = shape_[1];
            oss << "[";
            for (uint64_t r = 0; r < rows; ++r) {
                oss << "[";
                for (uint64_t c = 0; c < cols; ++c) {
                    oss << data_[r * cols + c];
                    if (c + 1 < cols) oss << ", ";
                }
                oss << "]";
                if (r + 1 < rows) oss << ", ";
            }
            oss << "]";
        } else {
            oss << "[";
            for (uint64_t i = 0; i < size_; ++i) {
                oss << data_[i];
                if (i + 1 < size_) oss << ", ";
            }
            oss << "]";
        }
        oss << ")";
        return oss.str();
    }

private:
    static uint64_t computeSize(const std::vector<uint64_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), static_cast<uint64_t>(1),
                                std::multiplies<uint64_t>());
    }

    void allocateBuffer() {
        if (size_ == 0) {
            data_ = nullptr;
            return;
        }
        data_ = static_cast<T*>(allocator_->allocate(static_cast<std::size_t>(size_) * sizeof(T)));
    }

    void releaseBuffer() {
        if (data_ != nullptr && allocator_ != nullptr) {
            allocator_->deallocate(data_, static_cast<std::size_t>(size_) * sizeof(T));
        }
        data_ = nullptr;
    }

    std::vector<uint64_t> shape_;
    T* data_ = nullptr;
    uint64_t size_ = 0;
    Allocator* allocator_ = &CpuAllocator::instance();
};

}
