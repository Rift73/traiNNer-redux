/*
 * Exact uniform and palette error-diffusion, Riemersma and palette quantize
 * dithering (CUDA).
 *
 * Bit-identical to chainner_ext 0.3.10's error_diffusion_dither, riemersma_dither
 * and quantize with UniformQuantization and PaletteQuantization. The float32
 * operation sequences follow chaiNNer-rs image-ops/dither (MIT, Copyright (c)
 * 2023 Michael Schmidt). Every product and sum is an explicit _rn intrinsic, so
 * nvcc cannot contract a multiply and an add into an FMA, which would change the
 * rounding.
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
 *
 * Palette paths: a palette couples the channels, so one block per image runs the
 * same wavefront over whole pixels (error diffusion), one warp per image walks
 * the traversal with the palette scan split across its lanes (Riemersma), and
 * palette quantize is one thread per pixel. Each image's palette (at most 256
 * colors, in PaletteQuantization's order) sits in shared memory.
 */

#include <torch/extension.h>
// rpcndr.h (windows.h via CUDA 13's nvtx3) has #define small char, which breaks c10's headers.
#ifdef _WIN32
#undef small
#endif
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
constexpr int kWarp = 32;
constexpr unsigned kFullWarp = 0xffffffffu;

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

// The algorithm's taps and the wavefront slope k: the smallest integer with
// dx + k*dy > 0 for every tap.
TapSet make_tap_set(int64_t algorithm, int& k) {
    TapSet set{};
    set.count = kTapCounts[algorithm];
    k = 1;
    for (int t = 0; t < set.count; ++t) {
        set.taps[t] = kTaps[algorithm][t];
        if (set.taps[t].dy > 0) k = std::max(k, -set.taps[t].dx / set.taps[t].dy + 1);
    }
    return set;
}

// Threads for the wavefront: whole warps up to the rows a step can hold.
int wavefront_threads(int height, int width, int k) {
    const int active = std::min(height, (width + k - 1) / k);
    return std::min(1024, (active + kWarp - 1) / kWarp * kWarp);
}

// PaletteQuantization's lookup below 300 colors (chaiNNer-rs ColorPalette): the
// squared distance summed from 0 in channel order, no FMA, and a scan with strict
// <, so the first entry reaching the minimum wins and entry 0 does when its
// distance is NaN. Palettes are entry-major: entry p's channel c at p*C + c.
template <int C>
__device__ __forceinline__ float palette_distance(
    const float* entry, const float (&color)[C]
) {
    float distance = 0.0f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
        const float delta = __fsub_rn(entry[c], color[c]);
        distance = __fadd_rn(distance, __fmul_rn(delta, delta));
    }
    return distance;
}

template <int C>
__device__ __forceinline__ int nearest_entry(
    const float* palette, int count, const float (&color)[C]
) {
    int best = 0;
    float best_distance = palette_distance<C>(palette, color);
    for (int p = 1; p < count; ++p) {
        const float distance = palette_distance<C>(palette + p * C, color);
        if (distance < best_distance) {
            best = p;
            best_distance = distance;
        }
    }
    return best;
}

// The scan's result from one warp: lane l scans entries l, l + 32, ... and keeps
// the first entry at its minimum, or none when it saw only NaN or +inf distances;
// the lanes' candidates combine to the smallest entry at the minimum over the
// finite distances. Entry 0 when its distance is NaN or no distance is finite,
// as the scan gives. Every lane returns the same entry.
template <int C>
__device__ __forceinline__ int warp_nearest_entry(
    const float* palette, int count, const float (&color)[C], int lane
) {
    int best = -1;
    float best_distance = __int_as_float(0x7f800000);
    for (int p = lane; p < count; p += kWarp) {
        const float distance = palette_distance<C>(palette + p * C, color);
        if (distance < best_distance) {
            best = p;
            best_distance = distance;
        }
    }
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        const int other = __shfl_down_sync(kFullWarp, best, offset);
        const float other_distance = __shfl_down_sync(kFullWarp, best_distance, offset);
        if (other >= 0 &&
            (best < 0 || other_distance < best_distance ||
             (other_distance == best_distance && other < best))) {
            best = other;
            best_distance = other_distance;
        }
    }
    best = __shfl_sync(kFullWarp, best, 0);
    return best < 0 || isnan(palette_distance<C>(palette, color)) ? 0 : best;
}

// The diffusing palette paths clip the adjusted color to [0, 1] before the
// lookup: glam's vector clamp (max, then min) maps NaN to 0; the scalar one for
// one channel keeps it.
template <int C>
__device__ __forceinline__ float palette_clip(float value) {
    if (C > 1 && isnan(value)) return 0.0f;
    if (value < 0.0f) return 0.0f;
    return value > 1.0f ? 1.0f : value;
}

// The block's image's palette (count entries of a padded (N, max_colors, C)
// table) into shared memory; the caller synchronizes.
template <int C>
__device__ __forceinline__ void load_palette(
    float* shared_palette, const float* palettes, int image, int max_colors, int count
) {
    const float* source = palettes + static_cast<int64_t>(image) * max_colors * C;
    for (int i = threadIdx.x; i < count * C; i += blockDim.x) shared_palette[i] = source[i];
}

// Palette quantize: the source pixel, unclipped, to its nearest palette entry.
// Grid (pixel blocks, images).
template <int C>
__global__ void palette_quantize_kernel(
    const float* __restrict__ src,
    float* __restrict__ out,
    const float* __restrict__ palettes,
    const int64_t* __restrict__ counts,
    const int max_colors,
    const int64_t pixels
) {
    extern __shared__ float shared_palette[];
    const int image = blockIdx.y;
    const int count = static_cast<int>(counts[image]);
    load_palette<C>(shared_palette, palettes, image, max_colors, count);
    __syncthreads();
    const int64_t image_offset = static_cast<int64_t>(image) * C * pixels;
    const float* image_src = src + image_offset;
    float* image_out = out + image_offset;
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < pixels;
         i += step) {
        float color[C];
#pragma unroll
        for (int c = 0; c < C; ++c) color[c] = image_src[c * pixels + i];
        const float* nearest =
            shared_palette + nearest_entry<C>(shared_palette, count, color) * C;
#pragma unroll
        for (int c = 0; c < C; ++c) image_out[c * pixels + i] = nearest[c];
    }
}

// Palette error diffusion: error_diffusion_kernel's wavefront over whole pixels,
// one block per image, the ring holding C floats per cell. The pushed error is
// the clipped color minus the nearest entry.
template <int C>
__global__ void palette_error_diffusion_kernel(
    const float* __restrict__ src,
    float* __restrict__ out,
    const float* __restrict__ palettes,
    const int64_t* __restrict__ counts,
    float* scratch,        // per-image rings when they exceed shared memory
    const int max_colors,
    const TapSet set,
    const int k,
    const int height,
    const int width
) {
    extern __shared__ float shared[];
    const int image = blockIdx.x;
    const int count = static_cast<int>(counts[image]);
    float* palette = shared;
    load_palette<C>(palette, palettes, image, max_colors, count);
    const int stride = width + 2 * kPad;
    const int ring_floats = kRingRows * stride * C;
    float* ring = scratch ? scratch + static_cast<int64_t>(image) * ring_floats
                          : shared + max_colors * C;
    for (int i = threadIdx.x; i < ring_floats; i += blockDim.x) ring[i] = 0.0f;

    const int64_t pixels = static_cast<int64_t>(height) * width;
    const int64_t image_offset = static_cast<int64_t>(image) * C * pixels;
    const float* image_src = src + image_offset;
    float* image_out = out + image_offset;
    const int steps = width + k * (height - 1);
    __syncthreads();

    for (int step = 0; step < steps; ++step) {
        const int first = step < width ? 0 : (step - width) / k + 1;
        const int last = min(height - 1, step / k);
        for (int y = first + static_cast<int>(threadIdx.x); y <= last; y += blockDim.x) {
            const int x = step - k * y;
            const int slot = y % kRingRows;
            float error[C];
#pragma unroll
            for (int c = 0; c < C; ++c) error[c] = 0.0f;
#pragma unroll
            for (int t = kMaxTaps - 1; t >= 0; --t) {
                if (t < set.count) {
                    const Tap tap = set.taps[t];
                    int row = slot - tap.dy;
                    row += row < 0 ? kRingRows : 0;
                    const float* source = ring + (row * stride + x - tap.dx + kPad) * C;
#pragma unroll
                    for (int c = 0; c < C; ++c) {
                        error[c] = __fadd_rn(error[c], __fmul_rn(source[c], tap.weight));
                    }
                }
            }
            const int64_t index = static_cast<int64_t>(y) * width + x;
            float color[C];
#pragma unroll
            for (int c = 0; c < C; ++c) {
                color[c] = palette_clip<C>(__fadd_rn(image_src[c * pixels + index], error[c]));
            }
            const float* nearest = palette + nearest_entry<C>(palette, count, color) * C;
            float* cell = ring + (slot * stride + x + kPad) * C;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                image_out[c * pixels + index] = nearest[c];
                cell[c] = __fsub_rn(color[c], nearest[c]);
            }
        }
        __syncthreads();
    }
}

// Palette Riemersma: one warp per image walks the pixels, pre-gathered into the
// traversal order, with the palette scan split across the lanes. Histories up to
// 32 keep slot l in lane l's registers (the error sum shuffles them in slot
// order); longer ones live in shared memory, or in a per-image global scratch
// beyond it. The history takes the original minus the nearest entry.
template <int C>
__global__ void palette_riemersma_kernel(
    const float* __restrict__ src,
    float* __restrict__ out,
    const float* __restrict__ palettes,
    const int64_t* __restrict__ counts,
    float* scratch,
    const int max_colors,
    const int64_t pixels,
    const int history,
    const float base
) {
    extern __shared__ float shared[];
    const int image = blockIdx.x;
    const int lane = static_cast<int>(threadIdx.x);
    const int count = static_cast<int>(counts[image]);
    float* palette = shared;
    load_palette<C>(palette, palettes, image, max_colors, count);
    const bool in_lanes = history <= kWarp;
    float* slots = in_lanes ? nullptr
                   : scratch ? scratch + static_cast<int64_t>(image) * history * C
                             : shared + max_colors * C;
    if (!in_lanes) {
        for (int i = lane; i < history * C; i += kWarp) slots[i] = 0.0f;
    }
    __syncthreads();

    const int64_t image_offset = static_cast<int64_t>(image) * C * pixels;
    const float* image_src = src + image_offset;
    float* image_out = out + image_offset;
    float mine[C];
#pragma unroll
    for (int c = 0; c < C; ++c) mine[c] = 0.0f;
    int slot = 0;
    for (int64_t i = 0; i < pixels; ++i) {
        float error[C];
#pragma unroll
        for (int c = 0; c < C; ++c) error[c] = 0.0f;
        if (in_lanes) {
            for (int h = 0; h < history; ++h) {
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    error[c] = __fadd_rn(error[c], __shfl_sync(kFullWarp, mine[c], h));
                }
            }
#pragma unroll
            for (int c = 0; c < C; ++c) mine[c] = __fmul_rn(mine[c], base);
        } else {
            for (int h = 0; h < history; ++h) {
#pragma unroll
                for (int c = 0; c < C; ++c) error[c] = __fadd_rn(error[c], slots[h * C + c]);
            }
            __syncwarp();
            for (int j = lane; j < history * C; j += kWarp) slots[j] = __fmul_rn(slots[j], base);
            __syncwarp();
        }
        float value[C], color[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            value[c] = image_src[c * pixels + i];
            color[c] = palette_clip<C>(__fadd_rn(value[c], error[c]));
        }
        const float* nearest = palette + warp_nearest_entry<C>(palette, count, color, lane) * C;
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < C; ++c) image_out[c * pixels + i] = nearest[c];
        }
        if (in_lanes) {
            if (lane == slot) {
#pragma unroll
                for (int c = 0; c < C; ++c) mine[c] = __fsub_rn(value[c], nearest[c]);
            }
        } else {
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < C; ++c) slots[slot * C + c] = __fsub_rn(value[c], nearest[c]);
            }
            __syncwarp();
        }
        slot = slot + 1 == history ? 0 : slot + 1;
    }
}

// Runs launch with std::integral_constant<int, C> for the image's channel count.
template <typename Launch>
void for_palette_channels(int64_t channels, Launch launch) {
    switch (channels) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        default: TORCH_CHECK(false, "palette dithering takes 1 or 3 channels, got ", channels);
    }
}

}  // namespace

torch::Tensor error_diffusion_cuda(
    const torch::Tensor& src, const torch::Tensor& levels, int64_t algorithm
) {
    const c10::cuda::CUDAGuard guard(src.device());
    const int channels = static_cast<int>(src.size(1));
    const int height = static_cast<int>(src.size(2));
    const int width = static_cast<int>(src.size(3));
    const int planes = static_cast<int>(src.size(0)) * channels;

    int k;
    const TapSet set = make_tap_set(algorithm, k);
    auto out = torch::empty_like(src);
    const int threads = wavefront_threads(height, width, k);
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

// palettes: (N, max_colors, C) in PaletteQuantization's order, counts: int64 of N.
torch::Tensor palette_quantize_cuda(
    const torch::Tensor& src, const torch::Tensor& palettes, const torch::Tensor& counts
) {
    const c10::cuda::CUDAGuard guard(src.device());
    const int64_t channels = src.size(1);
    const int64_t pixels = src.size(2) * src.size(3);
    const int max_colors = static_cast<int>(palettes.size(1));
    auto out = torch::empty_like(src);
    constexpr int threads = 256;
    const dim3 grid(
        static_cast<unsigned>(std::min<int64_t>((pixels + threads - 1) / threads, 1024)),
        static_cast<unsigned>(src.size(0))
    );
    const size_t shared_bytes = static_cast<size_t>(max_colors) * channels * sizeof(float);
    for_palette_channels(channels, [&](auto tag) {
        palette_quantize_kernel<decltype(tag)::value>
            <<<grid, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                src.data_ptr<float>(),
                out.data_ptr<float>(),
                palettes.data_ptr<float>(),
                counts.data_ptr<int64_t>(),
                max_colors,
                pixels
            );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor palette_error_diffusion_cuda(
    const torch::Tensor& src,
    const torch::Tensor& palettes,
    const torch::Tensor& counts,
    int64_t algorithm
) {
    const c10::cuda::CUDAGuard guard(src.device());
    const int64_t channels = src.size(1);
    const int height = static_cast<int>(src.size(2));
    const int width = static_cast<int>(src.size(3));
    const int images = static_cast<int>(src.size(0));
    const int max_colors = static_cast<int>(palettes.size(1));

    int k;
    const TapSet set = make_tap_set(algorithm, k);
    auto out = torch::empty_like(src);
    const int threads = wavefront_threads(height, width, k);
    const int64_t ring_floats = static_cast<int64_t>(kRingRows) * (width + 2 * kPad) * channels;
    const size_t palette_bytes = static_cast<size_t>(max_colors) * channels * sizeof(float);
    size_t shared_bytes = palette_bytes + ring_floats * sizeof(float);
    torch::Tensor scratch;
    if (shared_bytes > kSharedLimit) {
        scratch = torch::empty({images * ring_floats}, src.options());
        shared_bytes = palette_bytes;
    }
    for_palette_channels(channels, [&](auto tag) {
        palette_error_diffusion_kernel<decltype(tag)::value>
            <<<images, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                src.data_ptr<float>(),
                out.data_ptr<float>(),
                palettes.data_ptr<float>(),
                counts.data_ptr<int64_t>(),
                scratch.defined() ? scratch.data_ptr<float>() : nullptr,
                max_colors,
                set,
                k,
                height,
                width
            );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// src: (N, C, pixels) contiguous, each image already in traversal order.
torch::Tensor palette_riemersma_cuda(
    const torch::Tensor& src,
    const torch::Tensor& palettes,
    const torch::Tensor& counts,
    int64_t history,
    float base
) {
    const c10::cuda::CUDAGuard guard(src.device());
    const int64_t channels = src.size(1);
    const int images = static_cast<int>(src.size(0));
    const int64_t pixels = src.size(2);
    const int max_colors = static_cast<int>(palettes.size(1));

    auto out = torch::empty_like(src);
    const size_t palette_bytes = static_cast<size_t>(max_colors) * channels * sizeof(float);
    const int64_t history_floats = history > kWarp ? history * channels : 0;
    size_t shared_bytes = palette_bytes + history_floats * sizeof(float);
    torch::Tensor scratch;
    if (shared_bytes > kSharedLimit) {
        scratch = torch::empty({images * history_floats}, src.options());
        shared_bytes = palette_bytes;
    }
    for_palette_channels(channels, [&](auto tag) {
        palette_riemersma_kernel<decltype(tag)::value>
            <<<images, kWarp, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                src.data_ptr<float>(),
                out.data_ptr<float>(),
                palettes.data_ptr<float>(),
                counts.data_ptr<int64_t>(),
                scratch.defined() ? scratch.data_ptr<float>() : nullptr,
                max_colors,
                pixels,
                static_cast<int>(history),
                base
            );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
