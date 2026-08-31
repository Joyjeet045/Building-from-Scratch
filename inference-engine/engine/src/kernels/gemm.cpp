/*
 * gemm.cpp
 * Implementations of the GEMM kernels. Each strategy accumulates the raw
 * product into the output buffer and a shared finalize pass then applies
 * alpha, the broadcast bias scaled by beta, and the optional fused ReLU.
 */
#include "ie/kernels/gemm.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define IE_RESTRICT __restrict__
#else
#define IE_RESTRICT
#endif

namespace ie {

namespace {

constexpr uint64_t kTile = 64;
constexpr uint64_t kPackTile = 64;

struct Dims {
    uint64_t N;
    uint64_t K;
    uint64_t M;
};

Dims resolveDims(const Tensor<float>& A, const Tensor<float>& B, const GemmParams& params) {
    Dims d{};
    d.N = params.transA ? A.dim(1) : A.dim(0);
    d.K = params.transA ? A.dim(0) : A.dim(1);
    d.M = params.transB ? B.dim(0) : B.dim(1);
    return d;
}

void accumulateNaive(const float* IE_RESTRICT A, const float* IE_RESTRICT B, float* IE_RESTRICT C,
                      const Dims& d, const GemmParams& p, uint64_t rowBegin, uint64_t rowEnd) {
    for (uint64_t i = rowBegin; i < rowEnd; ++i) {
        for (uint64_t j = 0; j < d.M; ++j) {
            float acc = 0.0f;
            for (uint64_t k = 0; k < d.K; ++k) {
                float a = p.transA ? A[k * d.N + i] : A[i * d.K + k];
                float b = p.transB ? B[j * d.K + k] : B[k * d.M + j];
                acc += a * b;
            }
            C[i * d.M + j] = acc;
        }
    }
}

void accumulateDotContiguous(const float* IE_RESTRICT A, const float* IE_RESTRICT B, float* IE_RESTRICT C,
                              const Dims& d, uint64_t rowBegin, uint64_t rowEnd) {
    for (uint64_t i = rowBegin; i < rowEnd; ++i) {
        const float* arow = A + i * d.K;
        for (uint64_t j = 0; j < d.M; ++j) {
            const float* brow = B + j * d.K;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            uint64_t k = 0;
            for (; k + 4 <= d.K; k += 4) {
                a0 += arow[k] * brow[k];
                a1 += arow[k + 1] * brow[k + 1];
                a2 += arow[k + 2] * brow[k + 2];
                a3 += arow[k + 3] * brow[k + 3];
            }
            float acc = (a0 + a1) + (a2 + a3);
            for (; k < d.K; ++k) acc += arow[k] * brow[k];
            C[i * d.M + j] = acc;
        }
    }
}

void accumulateReordered(const float* IE_RESTRICT A, const float* IE_RESTRICT B, float* IE_RESTRICT C,
                          const Dims& d, const GemmParams& p, uint64_t rowBegin, uint64_t rowEnd) {
    if (p.transB && !p.transA) {
        accumulateDotContiguous(A, B, C, d, rowBegin, rowEnd);
        return;
    }
    for (uint64_t i = rowBegin; i < rowEnd; ++i) {
        float* crow = C + i * d.M;
        std::fill(crow, crow + d.M, 0.0f);
        for (uint64_t k = 0; k < d.K; ++k) {
            float a = p.transA ? A[k * d.N + i] : A[i * d.K + k];
            if (a == 0.0f) continue;
            if (p.transB) {
                for (uint64_t j = 0; j < d.M; ++j) crow[j] += a * B[j * d.K + k];
            } else {
                const float* brow = B + k * d.M;
                for (uint64_t j = 0; j < d.M; ++j) crow[j] += a * brow[j];
            }
        }
    }
}

void accumulateTiled(const float* IE_RESTRICT A, const float* IE_RESTRICT B, float* IE_RESTRICT C,
                      const Dims& d, const GemmParams& p, uint64_t rowBegin, uint64_t rowEnd) {
    for (uint64_t i = rowBegin; i < rowEnd; ++i) {
        std::fill(C + i * d.M, C + i * d.M + d.M, 0.0f);
    }
    for (uint64_t ii = rowBegin; ii < rowEnd; ii += kTile) {
        uint64_t iMax = std::min(ii + kTile, rowEnd);
        for (uint64_t kk = 0; kk < d.K; kk += kTile) {
            uint64_t kMax = std::min(kk + kTile, d.K);
            for (uint64_t jj = 0; jj < d.M; jj += kTile) {
                uint64_t jMax = std::min(jj + kTile, d.M);
                for (uint64_t i = ii; i < iMax; ++i) {
                    float* crow = C + i * d.M;
                    for (uint64_t k = kk; k < kMax; ++k) {
                        float a = p.transA ? A[k * d.N + i] : A[i * d.K + k];
                        if (a == 0.0f) continue;
                        if (p.transB) {
                            for (uint64_t j = jj; j < jMax; ++j) crow[j] += a * B[j * d.K + k];
                        } else {
                            const float* brow = B + k * d.M;
                            for (uint64_t j = jj; j < jMax; ++j) crow[j] += a * brow[j];
                        }
                    }
                }
            }
        }
    }
}

void microKernel(const float* IE_RESTRICT packA, const float* IE_RESTRICT packB, float* IE_RESTRICT C,
                  uint64_t rows, uint64_t depth, uint64_t cols, uint64_t ldc) {
    for (uint64_t i = 0; i < rows; ++i) {
        float* crow = C + i * ldc;
        const float* arow = packA + i * depth;
        for (uint64_t k = 0; k < depth; ++k) {
            float a = arow[k];
            if (a == 0.0f) continue;
            const float* brow = packB + k * cols;
            for (uint64_t j = 0; j < cols; ++j) crow[j] += a * brow[j];
        }
    }
}

void accumulatePacked(const float* IE_RESTRICT A, const float* IE_RESTRICT B, float* IE_RESTRICT C,
                       const Dims& d, const GemmParams& p, uint64_t rowBegin, uint64_t rowEnd) {
    for (uint64_t i = rowBegin; i < rowEnd; ++i) {
        std::fill(C + i * d.M, C + i * d.M + d.M, 0.0f);
    }

    static thread_local std::vector<float> packA;
    static thread_local std::vector<float> packB;
    packA.resize(kPackTile * kPackTile);
    packB.resize(kPackTile * kPackTile);

    for (uint64_t ii = rowBegin; ii < rowEnd; ii += kPackTile) {
        uint64_t iMax = std::min(ii + kPackTile, rowEnd);
        uint64_t rows = iMax - ii;
        for (uint64_t kk = 0; kk < d.K; kk += kPackTile) {
            uint64_t kMax = std::min(kk + kPackTile, d.K);
            uint64_t depth = kMax - kk;

            for (uint64_t i = 0; i < rows; ++i) {
                for (uint64_t k = 0; k < depth; ++k) {
                    packA[i * depth + k] =
                        p.transA ? A[(kk + k) * d.N + (ii + i)] : A[(ii + i) * d.K + (kk + k)];
                }
            }

            for (uint64_t jj = 0; jj < d.M; jj += kPackTile) {
                uint64_t jMax = std::min(jj + kPackTile, d.M);
                uint64_t cols = jMax - jj;

                for (uint64_t k = 0; k < depth; ++k) {
                    for (uint64_t j = 0; j < cols; ++j) {
                        packB[k * cols + j] =
                            p.transB ? B[(jj + j) * d.K + (kk + k)] : B[(kk + k) * d.M + (jj + j)];
                    }
                }

                microKernel(packA.data(), packB.data(), C + ii * d.M + jj, rows, depth, cols, d.M);
            }
        }
    }
}

void finalize(float* IE_RESTRICT C, const float* IE_RESTRICT bias, uint64_t biasSize, const Dims& d,
               const GemmParams& p, uint64_t rowBegin, uint64_t rowEnd) {
    bool perColumn = biasSize == d.M;
    bool full = biasSize == d.N * d.M;
    for (uint64_t i = rowBegin; i < rowEnd; ++i) {
        float* crow = C + i * d.M;
        for (uint64_t j = 0; j < d.M; ++j) {
            float value = p.alpha * crow[j];
            if (perColumn) {
                value += p.beta * bias[j];
            } else if (full) {
                value += p.beta * bias[i * d.M + j];
            }
            crow[j] = p.fuseRelu ? std::max(0.0f, value) : value;
        }
    }
}

GemmStrategy resolveAuto(const Dims& d, const GemmParams& p) {
    (void)p;
    uint64_t work = d.N * d.K * d.M;
    if (work < (1ull << 12)) return GemmStrategy::Reordered;
    if (d.N >= 16) return GemmStrategy::Packed;
    return GemmStrategy::Tiled;
}

}

GemmStrategy gemmStrategyFromString(const std::string& name) {
    if (name == "naive") return GemmStrategy::Naive;
    if (name == "reordered") return GemmStrategy::Reordered;
    if (name == "tiled") return GemmStrategy::Tiled;
    if (name == "packed") return GemmStrategy::Packed;
    if (name == "auto") return GemmStrategy::Auto;
    throw std::invalid_argument("Unknown gemm strategy: " + name);
}

const char* gemmStrategyName(GemmStrategy strategy) {
    switch (strategy) {
        case GemmStrategy::Naive: return "naive";
        case GemmStrategy::Reordered: return "reordered";
        case GemmStrategy::Tiled: return "tiled";
        case GemmStrategy::Packed: return "packed";
        case GemmStrategy::Auto: return "auto";
    }
    return "unknown";
}

void Gemm::validate(const Tensor<float>& A, const Tensor<float>& B, const Tensor<float>& bias,
                     const GemmParams& params) {
    if (A.rank() != 2 || B.rank() != 2) {
        throw std::invalid_argument("Gemm expects rank-2 tensors for A and B");
    }
    uint64_t innerA = params.transA ? A.dim(0) : A.dim(1);
    uint64_t innerB = params.transB ? B.dim(1) : B.dim(0);
    if (innerA != innerB) {
        throw std::invalid_argument("Gemm inner dimensions do not match");
    }
    Dims d = resolveDims(A, B, params);
    if (bias.size() != 0 && bias.size() != d.M && bias.size() != d.N * d.M) {
        throw std::invalid_argument("Gemm bias shape is not broadcastable to the output");
    }
}

std::vector<uint64_t> Gemm::outputShape(const Tensor<float>& A, const Tensor<float>& B,
                                         const GemmParams& params) {
    Dims d = resolveDims(A, B, params);
    return {d.N, d.M};
}

void Gemm::compute(const Tensor<float>& A, const Tensor<float>& B, const Tensor<float>& bias,
                    Tensor<float>& out, const GemmParams& params, GemmStrategy strategy, ThreadPool* pool) {
    validate(A, B, bias, params);
    Dims d = resolveDims(A, B, params);

    if (out.rank() != 2 || out.dim(0) != d.N || out.dim(1) != d.M) {
        out = Tensor<float>({d.N, d.M}, out.allocator());
    }

    if (strategy == GemmStrategy::Auto) strategy = resolveAuto(d, params);

    const float* aPtr = A.data();
    const float* bPtr = B.data();
    float* cPtr = out.data();
    const float* biasPtr = bias.data();
    uint64_t biasSize = bias.size();

    auto runRows = [&](std::size_t begin, std::size_t end) {
        uint64_t rowBegin = static_cast<uint64_t>(begin);
        uint64_t rowEnd = static_cast<uint64_t>(end);
        switch (strategy) {
            case GemmStrategy::Naive:
                accumulateNaive(aPtr, bPtr, cPtr, d, params, rowBegin, rowEnd);
                break;
            case GemmStrategy::Reordered:
                accumulateReordered(aPtr, bPtr, cPtr, d, params, rowBegin, rowEnd);
                break;
            case GemmStrategy::Tiled:
                accumulateTiled(aPtr, bPtr, cPtr, d, params, rowBegin, rowEnd);
                break;
            case GemmStrategy::Packed:
                accumulatePacked(aPtr, bPtr, cPtr, d, params, rowBegin, rowEnd);
                break;
            case GemmStrategy::Auto:
                break;
        }
        finalize(cPtr, biasPtr, biasSize, d, params, rowBegin, rowEnd);
    };

    uint64_t workPerRow = d.K * d.M;
    bool worthSharding = pool != nullptr && d.N > 1 && workPerRow * d.N >= (1ull << 18);
    if (worthSharding) {
        pool->parallelFor(static_cast<std::size_t>(d.N), runRows, 1);
    } else {
        runRows(0, static_cast<std::size_t>(d.N));
    }
}

}
