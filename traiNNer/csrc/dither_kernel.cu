/*
 * Exact uniform error-diffusion and Riemersma dithering (CUDA).
 *
 * Bit-identical to chainner_ext 0.3.10's error_diffusion_dither and
 * riemersma_dither with UniformQuantization. The float32 operation sequences
 * follow chaiNNer-rs image-ops/dither (MIT, Copyright (c) 2023 Michael Schmidt).
 * Every product and sum is an explicit _rn intrinsic, so nvcc cannot contract a
 * multiply and an add into an FMA, which would change the rounding.
 *
 * Error diffusion: one block per (image, channel) plane; uniform channels keep
 * independent error histories. The serial scan pushes each pixel's error into
 * later cells, so a cell's error is the sum, from 0 and in its sources' raster
 * order, of source error * weight. The kernel gathers that same sum: pixel (y, x)
 * runs at step x + k*y, where k is the smallest integer with dx + k*dy > 0 for
 * every tap (dy, dx), so every source of a pixel ran in an earlier step. Errors
 * live in a 3-row ring padded by 2 zero columns; every table has
 * k*(3 - dy) > dx, so no slot is overwritten before its last reader has run.
 *
 * Riemersma: one thread per plane walks its plane, pre-gathered into the
 * traversal order (built on the host), keeping the history ring in registers for
 * histories up to 32 and in shared memory beyond.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

namespace {

constexpr int kMaxTaps = 12;
constexpr int kRingRows = 3;
constexpr int kPad = 2;
constexpr int kMaxRegisterHistory = 32;
constexpr size_t kSharedLimit = 48 * 1024;

struct Tap {
    int dy, dx;
    float weight;
};

// chainner_ext's tables in DiffusionAlgorithm order, each sorted by (dy, dx).
constexpr Tap kTaps[8][kMaxTaps] = {
    {{0, 1, 0.4375f}, {1, -1, 0.1875f}, {1, 0, 0.3125f}, {1, 1, 0.0625f}},
    {{0, 1, 0.145833328f}, {0, 2, 0.104166664f}, {1, -2, 0.0625f}, {1, -1, 0.104166664f},
     {1, 0, 0.145833328f}, {1, 1, 0.104166664f}, {1, 2, 0.0625f}, {2, -2, 0.020833334f},
     {2, -1, 0.0625f}, {2, 0, 0.104166664f}, {2, 1, 0.0625f}, {2, 2, 0.020833334f}},
    {{0, 1, 0.190476194f}, {0, 2, 0.095238097f}, {1, -2, 0.0476190485f}, {1, -1, 0.095238097f},
     {1, 0, 0.190476194f}, {1, 1, 0.095238097f}, {1, 2, 0.0476190485f}, {2, -2, 0.0238095243f},
     {2, -1, 0.0476190485f}, {2, 0, 0.095238097f}, {2, 1, 0.0476190485f}, {2, 2, 0.0238095243f}},
    {{0, 1, 0.125f}, {0, 2, 0.125f}, {1, -1, 0.125f}, {1, 0, 0.125f}, {1, 1, 0.125f}, {2, 0, 0.125f}},
    {{0, 1, 0.25f}, {0, 2, 0.125f}, {1, -2, 0.0625f}, {1, -1, 0.125f}, {1, 0, 0.25f}, {1, 1, 0.125f},
     {1, 2, 0.0625f}},
    {{0, 1, 0.15625f}, {0, 2, 0.09375f}, {1, -2, 0.0625f}, {1, -1, 0.125f}, {1, 0, 0.15625f},
     {1, 1, 0.125f}, {1, 2, 0.0625f}, {2, -1, 0.0625f}, {2, 0, 0.09375f}, {2, 1, 0.0625f}},
    {{0, 1, 0.25f}, {0, 2, 0.1875f}, {1, -2, 0.0625f}, {1, -1, 0.125f}, {1, 0, 0.1875f},
     {1, 1, 0.125f}, {1, 2, 0.0625f}},
    {{0, 1, 0.5f}, {1, -1, 0.25f}, {1, 0, 0.25f}}};
constexpr int kTapCounts[8] = {4, 12, 12, 6, 7, 10, 7, 3};

// Passed by value (kernel parameter space); fully unrolled loops keep the
// indices static, so the table never spills to local memory.
struct TapSet {
    Tap taps[kMaxTaps];
    int count;
};

// The reference's nearest level: glam's scalar round (half up) for one channel,
// its vector round otherwise. clip keeps NaN; only the vector path maps NaN to 0.
//
// glam's vector round is half to even computed as floor(v) + 1 when rounding up
// (chainner_ext's vector_round), so v in [-0.5, 0) gives +0 where rintf gives
// -0; for every other v but -0 the two agree. Adding +0 maps that -0 to +0. v is
// never -0 here: color = value + error, where error is a sum folded from +0.0f
// and so never -0, and a float sum is -0 only when both terms are -0; factor >= 1.
__device__ __forceinline__ float nearest_level(
    float color, float factor, float inverse, bool single
) {
    float value = __fmul_rn(color, factor);
    value = single ? floorf(__fadd_rn(value, 0.5f)) : __fadd_rn(rintf(value), 0.0f);
    value = __fmul_rn(value, inverse);
    if (!single && isnan(value)) return 0.0f;
    if (value < 0.0f) return 0.0f;
    return value > 1.0f ? 1.0f : value;
}

__global__ void error_diffusion_kernel(
    const float* __restrict__ src,
    float* __restrict__ out,
    const int64_t* __restrict__ levels,
    float* scratch,        // per-plane rings when they exceed shared memory
    const TapSet set,
    const int k,
    const int channels,
    const int height,
    const int width
) {
    extern __shared__ float shared_ring[];
    const int plane = blockIdx.x;
    const int64_t plane_offset = static_cast<int64_t>(plane) * height * width;
    const float* plane_src = src + plane_offset;
    float* plane_out = out + plane_offset;
    const int stride = width + 2 * kPad;
    float* ring = scratch ? scratch + static_cast<int64_t>(plane) * kRingRows * stride
                          : shared_ring;
    for (int i = threadIdx.x; i < kRingRows * stride; i += blockDim.x) ring[i] = 0.0f;

    const float factor =
        static_cast<float>(static_cast<uint32_t>(levels[plane / channels]) - 1u);
    const float inverse = __fdiv_rn(1.0f, factor);
    const bool single = channels == 1;
    const int steps = width + k * (height - 1);
    __syncthreads();

    for (int step = 0; step < steps; ++step) {
        // Rows whose pixel x = step - k*y lies in [0, width).
        const int first = step < width ? 0 : (step - width) / k + 1;
        const int last = min(height - 1, step / k);
        for (int y = first + static_cast<int>(threadIdx.x); y <= last; y += blockDim.x) {
            const int x = step - k * y;
            const int slot = y % kRingRows;
            float error = 0.0f;
            // Reverse table order = (dy, dx) descending = the sources' raster order.
#pragma unroll
            for (int t = kMaxTaps - 1; t >= 0; --t) {
                if (t < set.count) {
                    const Tap tap = set.taps[t];
                    int row = slot - tap.dy;
                    row += row < 0 ? kRingRows : 0;
                    const float source = ring[row * stride + x - tap.dx + kPad];
                    error = __fadd_rn(error, __fmul_rn(source, tap.weight));
                }
            }
            const int64_t index = static_cast<int64_t>(y) * width + x;
            const float color = __fadd_rn(plane_src[index], error);
            const float nearest = nearest_level(color, factor, inverse, single);
            plane_out[index] = nearest;
            ring[slot * stride + x + kPad] = __fsub_rn(color, nearest);
        }
        __syncthreads();
    }
}

// One Riemersma pixel on a register history: sum in slot order, decay every
// slot, then store original minus nearest (not the adjusted color) in slot j.
template <int History>
__device__ __forceinline__ float riemersma_pixel(
    float (&slots)[History], int j, float value, float base, float factor, float inverse,
    bool single
) {
    float error = 0.0f;
#pragma unroll
    for (int k = 0; k < History; ++k) {
        error = __fadd_rn(error, slots[k]);
        slots[k] = __fmul_rn(slots[k], base);
    }
    const float nearest = nearest_level(__fadd_rn(value, error), factor, inverse, single);
    slots[j] = __fsub_rn(value, nearest);
    return nearest;
}

// Riemersma with the history ring in registers; one single-thread block per
// plane, so no two serial chains share a lockstep warp whose loads and stores
// would fan out across lanes. Planes arrive in traversal order. The pixel loop
// runs in chunks of History pixels, so pixel start + j always writes slot j and
// every register index is static. Full chunks carry no per-pixel guard, so the
// scheduler can overlap the next pixel's history sum with this pixel's
// quantization; the next chunk's loads overlap this chunk.
template <int History>
__global__ void riemersma_register_kernel(
    const float* __restrict__ src,
    float* __restrict__ out,
    const int64_t* __restrict__ levels,
    const int channels,
    const int64_t pixels,
    const float base
) {
    const int plane = blockIdx.x;
    const float* plane_src = src + plane * pixels;
    float* plane_out = out + plane * pixels;
    const float factor =
        static_cast<float>(static_cast<uint32_t>(levels[plane / channels]) - 1u);
    const float inverse = __fdiv_rn(1.0f, factor);
    const bool single = channels == 1;

    float slots[History], values[History], next[History];
#pragma unroll
    for (int j = 0; j < History; ++j) {
        slots[j] = 0.0f;
        values[j] = j < pixels ? plane_src[j] : 0.0f;
    }
    int64_t start = 0;
    for (; start + History <= pixels; start += History) {
#pragma unroll
        for (int j = 0; j < History; ++j) {
            const int64_t ahead = start + History + j;
            next[j] = ahead < pixels ? plane_src[ahead] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < History; ++j) {
            plane_out[start + j] =
                riemersma_pixel(slots, j, values[j], base, factor, inverse, single);
        }
#pragma unroll
        for (int j = 0; j < History; ++j) values[j] = next[j];
    }
    // The last, partial chunk; values already holds it.
#pragma unroll
    for (int j = 0; j < History; ++j) {
        if (start + j < pixels) {
            plane_out[start + j] =
                riemersma_pixel(slots, j, values[j], base, factor, inverse, single);
        }
    }
}

// Riemersma for histories beyond the register kernels, one single-thread block
// per plane: the ring lives in shared memory, or in a per-plane global scratch
// when it exceeds shared memory.
__global__ void riemersma_kernel(
    const float* __restrict__ src,
    float* __restrict__ out,
    const int64_t* __restrict__ levels,
    float* scratch,
    const int channels,
    const int64_t pixels,
    const int history,
    const float base
) {
    extern __shared__ float shared_history[];
    const int plane = blockIdx.x;
    float* slots = scratch ? scratch + static_cast<int64_t>(plane) * history : shared_history;
    for (int h = 0; h < history; ++h) slots[h] = 0.0f;

    const float* plane_src = src + plane * pixels;
    float* plane_out = out + plane * pixels;
    const float factor =
        static_cast<float>(static_cast<uint32_t>(levels[plane / channels]) - 1u);
    const float inverse = __fdiv_rn(1.0f, factor);
    const bool single = channels == 1;

    int slot = 0;
    for (int64_t i = 0; i < pixels; ++i) {
        float error = 0.0f;
        for (int h = 0; h < history; ++h) error = __fadd_rn(error, slots[h]);
        for (int h = 0; h < history; ++h) slots[h] = __fmul_rn(slots[h], base);
        const float value = plane_src[i];
        const float nearest = nearest_level(__fadd_rn(value, error), factor, inverse, single);
        plane_out[i] = nearest;
        slots[slot] = __fsub_rn(value, nearest);
        slot = slot + 1 == history ? 0 : slot + 1;
    }
}

struct RiemersmaLaunch {
    const float* src;
    float* out;
    const int64_t* levels;
    int planes, channels;
    int64_t pixels;
    float base;
    cudaStream_t stream;
};

template <int History>
void launch_riemersma_registers(const RiemersmaLaunch& l) {
    riemersma_register_kernel<History><<<l.planes, 1, 0, l.stream>>>(
        l.src, l.out, l.levels, l.channels, l.pixels, l.base
    );
}

template <int... Offsets>
constexpr std::array<void (*)(const RiemersmaLaunch&), sizeof...(Offsets)>
register_launchers(std::integer_sequence<int, Offsets...>) {
    return {&launch_riemersma_registers<Offsets + 2>...};
}

// Indexed by history - 2, for histories 2..kMaxRegisterHistory.
constexpr auto kRegisterLaunchers =
    register_launchers(std::make_integer_sequence<int, kMaxRegisterHistory - 1>{});

}  // namespace

torch::Tensor error_diffusion_cuda(
    const torch::Tensor& src, const torch::Tensor& levels, int64_t algorithm
) {
    const c10::cuda::CUDAGuard guard(src.device());
    const int channels = static_cast<int>(src.size(1));
    const int height = static_cast<int>(src.size(2));
    const int width = static_cast<int>(src.size(3));
    const int planes = static_cast<int>(src.size(0)) * channels;

    TapSet set{};
    set.count = kTapCounts[algorithm];
    int k = 1;
    for (int t = 0; t < set.count; ++t) {
        set.taps[t] = kTaps[algorithm][t];
        if (set.taps[t].dy > 0) k = std::max(k, -set.taps[t].dx / set.taps[t].dy + 1);
    }

    auto out = torch::empty_like(src);
    const int active = std::min(height, (width + k - 1) / k);
    const int threads = std::min(1024, (active + 31) / 32 * 32);
    const int64_t ring_floats = static_cast<int64_t>(kRingRows) * (width + 2 * kPad);
    size_t shared_bytes = ring_floats * sizeof(float);
    torch::Tensor scratch;
    if (shared_bytes > kSharedLimit) {
        scratch = torch::empty({planes * ring_floats}, src.options());
        shared_bytes = 0;
    }
    error_diffusion_kernel<<<planes, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<float>(),
        out.data_ptr<float>(),
        levels.data_ptr<int64_t>(),
        scratch.defined() ? scratch.data_ptr<float>() : nullptr,
        set,
        k,
        channels,
        height,
        width
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// src: (N, C, pixels) contiguous, each plane already in traversal order.
torch::Tensor riemersma_cuda(
    const torch::Tensor& src, const torch::Tensor& levels, int64_t history, float base
) {
    const c10::cuda::CUDAGuard guard(src.device());
    const int channels = static_cast<int>(src.size(1));
    const int planes = static_cast<int>(src.size(0)) * channels;
    const int64_t pixels = src.size(2);

    auto out = torch::empty_like(src);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (history <= kMaxRegisterHistory) {
        const RiemersmaLaunch launch{
            src.data_ptr<float>(), out.data_ptr<float>(), levels.data_ptr<int64_t>(),
            planes, channels, pixels, base, stream};
        kRegisterLaunchers[history - 2](launch);
    } else {
        size_t shared_bytes = static_cast<size_t>(history) * sizeof(float);
        torch::Tensor scratch;
        if (shared_bytes > kSharedLimit) {
            scratch = torch::empty({static_cast<int64_t>(planes) * history}, src.options());
            shared_bytes = 0;
        }
        riemersma_kernel<<<planes, 1, shared_bytes, stream>>>(
            src.data_ptr<float>(),
            out.data_ptr<float>(),
            levels.data_ptr<int64_t>(),
            scratch.defined() ? scratch.data_ptr<float>() : nullptr,
            channels,
            pixels,
            static_cast<int>(history),
            base
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
