/*
 * PyTorch C++ binding for the exact CUDA dithering kernels.
 *
 * Validates inputs, computes Riemersma's decay base on the host (glibc logf and
 * expf, as chainner_ext's Rust build calls them), and builds chainner_ext's
 * rectangular Hilbert traversal order for riemersma_dither.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

// Implemented in dither_kernel.cu
torch::Tensor error_diffusion_cuda(
    const torch::Tensor& src, const torch::Tensor& levels, int64_t algorithm
);
torch::Tensor riemersma_cuda(
    const torch::Tensor& src, const torch::Tensor& levels, int64_t history, float base
);

namespace {

constexpr int64_t kMaxPixels = std::numeric_limits<int32_t>::max();

void check_input(const torch::Tensor& src, const torch::Tensor& levels) {
    TORCH_CHECK(
        src.is_cuda() && src.scalar_type() == torch::kFloat32 && src.dim() == 4 &&
            src.is_contiguous(),
        "src must be a contiguous float32 CUDA tensor (N, C, H, W)"
    );
    const int64_t channels = src.size(1);
    TORCH_CHECK(
        channels == 1 || channels == 3 || channels == 4,
        "dithering supports 1, 3 or 4 channels, got ", channels
    );
    TORCH_CHECK(src.numel() > 0, "src must not be empty");
    TORCH_CHECK(src.size(2) * src.size(3) <= kMaxPixels, "image has too many pixels");
    TORCH_CHECK(
        levels.device() == src.device() && levels.scalar_type() == torch::kInt64 &&
            levels.dim() == 1 && levels.size(0) == src.size(0) && levels.is_contiguous(),
        "levels must be a contiguous int64 tensor of N on the input's device"
    );
}

/* chainner_ext's rectangular Hilbert traversal: zhang_hilbert 0.1.1 (src/core.rs,
   src/arb.rs; MIT, Copyright 2017 yvt), iterated without recursion or an
   image-sized coordinate list. */
struct HilbertLevel {
    size_t size[2];
    unsigned curve, progress;
};

struct HilbertScan {
    size_t size[2], levels, last, position[2], progress[2];
    HilbertLevel level[64];
    unsigned curve, end;
    bool secondary_negative, helper, done;
};

constexpr unsigned kCurveAddress[8] = {180, 120, 75, 135, 30, 45, 225, 210};
constexpr unsigned kCurveInduction[8][4] = {
    {1, 0, 0, 3}, {0, 1, 1, 2}, {3, 2, 2, 1}, {2, 3, 3, 0},
    {7, 4, 4, 5}, {6, 5, 5, 4}, {5, 6, 6, 7}, {4, 7, 7, 6}};

size_t integer_log2(size_t value) {
    size_t result = 0;
    while (value >>= 1) ++result;
    return result;
}

size_t hilbert_division(size_t size) {
    const size_t mask = size_t{1} << (integer_log2(size) - 1);
    return (size & mask) + mask;
}

void hilbert_extra(
    const size_t size[2], unsigned position, unsigned curve, size_t result[2]
) {
    position ^= (curve == 0 || curve == 5) ? 1u : 0u;
    for (size_t axis = 0; axis < 2; ++axis) {
        const size_t larger = ((size[axis] + 3) >> 2) << 1;
        result[axis] = (position & (2u >> axis)) ? larger : size[axis] - larger;
    }
}

void hilbert_block(HilbertScan* scan, unsigned curve, const size_t size[2]) {
    scan->secondary_negative = (curve & 2) != 0;
    scan->curve = curve;
    scan->end = kCurveAddress[curve] >> 6;
    scan->progress[0] = size[curve & 1];
    scan->progress[1] = size[(curve & 1) ^ 1];
}

void hilbert_init(HilbertScan* scan, size_t width, size_t height) {
    *scan = HilbertScan{};
    scan->size[0] = width;
    scan->size[1] = height;
    scan->levels = 1;
    if (width == 1 || height == 1) {
        scan->progress[0] = 1;
        scan->progress[1] = width == 1 ? height : width;
        scan->curve = width == 1 ? 0 : 1;
        return;
    }
    scan->levels = integer_log2(std::min(width, height)) + 1;
    scan->level[0].size[0] = width;
    scan->level[0].size[1] = height;
    for (size_t i = 1; i <= scan->levels - 2; ++i) {
        for (size_t axis = 0; axis < 2; ++axis) {
            const size_t previous = scan->level[i - 1].size[axis];
            scan->level[i].size[axis] = previous - hilbert_division(previous);
        }
        scan->level[i].curve = static_cast<unsigned>(i % 2);
    }
    scan->last = scan->levels - 2;
    unsigned curve;
    if (width & 1) {
        curve = 0;
        scan->helper = true;
    } else if (height & 1) {
        curve = scan->levels == 2 ? 0 : 1;
        scan->helper = scan->levels != 2;
    } else {
        curve = static_cast<unsigned>(scan->last % 2);
    }
    HilbertLevel* last = scan->level + scan->last;
    if (scan->helper) --last->size[curve & 1];
    last->curve = curve;
    size_t size[2] = {last->size[0], last->size[1]};
    if (size[0] >= 3 && size[1] >= 3) {
        hilbert_extra(last->size, 0, curve, size);
        curve = kCurveInduction[curve][0];
        ++scan->last;
        std::copy(size, size + 2, scan->level[scan->last].size);
    }
    hilbert_block(scan, curve, size);
}

void hilbert_primary_step(HilbertScan* scan, size_t axis) {
    if ((scan->curve ^ (scan->curve >> 1)) & 2) {
        --scan->position[axis];
    } else {
        ++scan->position[axis];
    }
}

bool hilbert_next(HilbertScan* scan, size_t position[2]) {
    if (scan->done) return false;
    std::copy(scan->position, scan->position + 2, position);
    size_t primary = scan->progress[0];
    size_t secondary = scan->progress[1] - 1;
    const size_t primary_axis = scan->curve & 1;
    const size_t secondary_axis = primary_axis ^ 1;
    if (secondary) {
        if (scan->secondary_negative) {
            --scan->position[secondary_axis];
        } else {
            ++scan->position[secondary_axis];
        }
        scan->progress[1] = secondary;
        return true;
    }
    --primary;
    secondary = scan->level[scan->last].size[secondary_axis];
    scan->secondary_negative = !scan->secondary_negative;
    if (primary) {
        hilbert_primary_step(scan, primary_axis);
        scan->progress[0] = primary;
        scan->progress[1] = secondary;
        return true;
    }
    if (scan->helper && (scan->last == scan->levels - 2 ||
                         scan->level[scan->levels - 2].progress == 3)) {
        HilbertLevel* level = scan->level + scan->levels - 2;
        const size_t axis = level->curve & 1;
        scan->end = 3;
        scan->curve = level->curve;
        scan->secondary_negative = false;
        scan->progress[0] = 1;
        scan->progress[1] = level->size[axis ^ 1];
        scan->helper = false;
        scan->last = scan->levels - 2;
        hilbert_primary_step(scan, axis);
        return true;
    }
    if (!scan->last) {
        scan->done = true;
        return true;
    }
    size_t i = scan->last - 1;
    unsigned enter = 0;
    for (;;) {
        HilbertLevel* level = scan->level + i;
        if (++level->progress == 4) {
            if (!i) {
                scan->done = true;
                return true;
            }
            --i;
        } else {
            const unsigned address = kCurveAddress[level->curve] >> (level->progress * 2 - 2);
            const unsigned relative = address ^ (address >> 2);
            if ((relative >> secondary_axis) & 1) {
                hilbert_primary_step(scan, primary_axis);
            } else if (scan->secondary_negative) {
                ++scan->position[secondary_axis];
            } else {
                --scan->position[secondary_axis];
            }
            enter = scan->end ^ (relative & 3);
            break;
        }
    }
    if (i == scan->levels - 2) {
        HilbertLevel* level = scan->level + i;
        const unsigned address = kCurveAddress[level->curve] >> (level->progress * 2);
        const unsigned curve = kCurveInduction[level->curve][level->progress];
        hilbert_extra(level->size, address, level->curve, scan->level[i + 1].size);
        hilbert_block(scan, curve, scan->level[i + 1].size);
        return true;
    }
    while (i < scan->levels - 2) {
        HilbertLevel* level = scan->level + i;
        HilbertLevel* next = level + 1;
        const unsigned address = kCurveAddress[level->curve] >> (level->progress * 2);
        for (size_t axis = 0; axis < 2; ++axis) {
            const size_t larger = hilbert_division(level->size[axis]);
            next->size[axis] =
                (address & (2u >> axis)) ? larger : level->size[axis] - larger;
        }
        next->curve = kCurveInduction[level->curve][level->progress];
        next->progress = 0;
        ++i;
    }
    size_t size[2] = {scan->level[i].size[0], scan->level[i].size[1]};
    const unsigned parity = static_cast<unsigned>((size[0] & 1) * 2 + (size[1] & 1));
    unsigned curve;
    bool helper = false;
    if (!parity) {
        static constexpr unsigned kScanningType[2][4][2] = {
            {{0, 1}, {6, 6}, {7, 7}, {3, 2}}, {{1, 0}, {5, 5}, {4, 4}, {2, 3}}};
        unsigned direction = 0;
        unsigned negative = 0;
        size_t parent = i - 1;
        for (;;) {
            const HilbertLevel* level = scan->level + parent;
            if (level->progress == 3) {
                if (!parent) break;
                --parent;
            } else {
                const unsigned address =
                    kCurveAddress[level->curve] >> (level->progress * 2);
                const unsigned relative = address ^ (address >> 2);
                direction = relative & 1;
                negative = (address & relative & 3) != 0;
                break;
            }
        }
        curve = kScanningType[negative][enter][direction];
    } else if (parity == 1) {
        helper = scan->position[0] + size[0] == scan->size[0] &&
                 scan->position[1] + 1 == size[1];
        curve = helper ? 5 : 6;
    } else {
        curve = 7;
    }
    if (helper) {
        --size[1];
        std::copy(size, size + 2, scan->level[i].size);
    }
    scan->level[i].curve = curve;
    if (size[0] >= 3 && size[1] >= 3) {
        scan->level[i].progress = 0;
        hilbert_extra(scan->level[i].size, enter, curve, size);
        curve = kCurveInduction[curve][0];
        ++i;
        std::copy(size, size + 2, scan->level[i].size);
    }
    hilbert_block(scan, curve, size);
    scan->helper = helper;
    scan->last = i;
    return true;
}

// Width of the next near-square part along the major axis.
size_t hilbert_part(size_t remaining, size_t minor) {
    size_t count = 1;
    if (remaining > minor) {
        const size_t k = remaining / minor;
        const size_t first = remaining / k - minor;
        const size_t second = minor - remaining / (k + 1);
        count = first < second ? k : k + 1;
    }
    if (count == 1) return remaining;
    const size_t width = remaining / count;
    return width + (width & 1);
}

}  // namespace

// riemersma_dither's visiting order as linear indices y * width + x, and its
// inverse permutation (both CPU int32).
std::tuple<torch::Tensor, torch::Tensor> riemersma_order(int64_t height, int64_t width) {
    TORCH_CHECK(height > 0 && width > 0, "image must not be empty");
    TORCH_CHECK(height * width <= kMaxPixels, "image has too many pixels");
    auto order = torch::empty({height * width}, torch::kInt32);
    auto inverse = torch::full({height * width}, -1, torch::kInt32);
    int32_t* data = order.data_ptr<int32_t>();
    int32_t* rank = inverse.data_ptr<int32_t>();
    const size_t h = static_cast<size_t>(height);
    const size_t w = static_cast<size_t>(width);
    const size_t major = std::max(w, h);
    const size_t minor = std::min(w, h);
    const bool transpose = h > w;
    int64_t count = 0;
    for (size_t start = 0; start < major;) {
        const size_t length = hilbert_part(major - start, minor);
        HilbertScan scan;
        hilbert_init(&scan, length, minor);
        size_t position[2];
        while (hilbert_next(&scan, position)) {
            const size_t x = transpose ? position[1] : position[0] + start;
            const size_t y = transpose ? position[0] + start : position[1];
            TORCH_CHECK(
                x < w && y < h && count < height * width,
                "Hilbert traversal left the image"
            );
            const size_t index = y * w + x;
            TORCH_CHECK(rank[index] < 0, "Hilbert traversal revisited a pixel");
            rank[index] = static_cast<int32_t>(count);
            data[count++] = static_cast<int32_t>(index);
        }
        start += length;
    }
    TORCH_CHECK(count == height * width, "Hilbert traversal missed pixels");
    return {order, inverse};
}

torch::Tensor error_diffusion(torch::Tensor src, torch::Tensor levels, int64_t algorithm) {
    check_input(src, levels);
    TORCH_CHECK(algorithm >= 0 && algorithm < 8, "algorithm must be in [0, 8)");
    return error_diffusion_cuda(src, levels, algorithm);
}

torch::Tensor riemersma(
    torch::Tensor src,
    torch::Tensor levels,
    torch::Tensor order,
    torch::Tensor inverse,
    int64_t history,
    double decay_ratio
) {
    check_input(src, levels);
    TORCH_CHECK(history >= 2, "Argument 'history_length' must be at least 2.");
    TORCH_CHECK(
        history <= std::numeric_limits<int32_t>::max() / 32, "history_length is too large"
    );
    const int64_t pixels = src.size(2) * src.size(3);
    for (const auto& permutation : {order, inverse}) {
        TORCH_CHECK(
            permutation.device() == src.device() &&
                permutation.scalar_type() == torch::kInt32 && permutation.dim() == 1 &&
                permutation.size(0) == pixels,
            "order and inverse must be riemersma_order(H, W) on the input's device"
        );
    }
    // riemersma_dither's float32 base and its assert.
    const float decay = static_cast<float>(decay_ratio);
    const float base = std::exp(std::log(decay) / (static_cast<float>(history) - 1.0f));
    TORCH_CHECK(base > 0.0f && base < 1.0f, "assertion failed: 0.0 < base && base < 1.0");
    // Gather each plane into traversal order so the serial walk reads sequentially,
    // then scatter the result back with the inverse permutation.
    const auto planes = src.view({src.size(0), src.size(1), pixels}).index_select(2, order);
    return riemersma_cuda(planes, levels, history, base).index_select(2, inverse).view_as(src);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("error_diffusion", &error_diffusion,
          "Uniform error-diffusion dithering (CUDA), bit-exact to chainner_ext",
          py::arg("src"), py::arg("levels"), py::arg("algorithm"));
    m.def("riemersma", &riemersma,
          "Uniform Riemersma dithering (CUDA), bit-exact to chainner_ext",
          py::arg("src"), py::arg("levels"), py::arg("order"), py::arg("inverse"),
          py::arg("history"), py::arg("decay_ratio"));
    m.def("riemersma_order", &riemersma_order,
          "riemersma_dither's traversal and its inverse as linear indices (CPU int32)",
          py::arg("height"), py::arg("width"));
}
