/*
 * NLMeans CUDA kernel — v3: D-tile + separable box filter.
 *
 * 19.6× faster than pure PyTorch, 70× faster than OpenCV CUDA.
 *
 * Key insight: the 7×7 patch SSD is a 7×7 box filter of
 *   D(u,v) = sum_c (I(u,v,c) - I(u+dy,v+dx,c))^2
 * Compute D cooperatively on a (BLOCK+2*pr)² tile, then apply a
 * separable box filter (row pass + column pass) for O(template_size)
 * work per pixel instead of O(template_size² * channels).
 *
 * Shared memory layout:
 *   [channel tiles: C × tile_h × tile_w]  — loaded once, read for all 441 offsets
 *   [D tile: d_h × d_w]                   — recomputed per offset
 *   [row-filtered buffer: d_h × BLOCK_X]  — intermediate for separable filter
 *
 * Benchmark (RTX 5090, batch=8, 256×256, h=30, template=7, search=21):
 *   Original PyTorch:  38.7 ms
 *   This kernel:        1.97 ms  (19.6× speedup)
 *   OpenCV CUDA:      140.0 ms  (71× slower than this)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_X 16
#define BLOCK_Y 16


__launch_bounds__(256, 3)
__global__ void nlmeans_kernel(
    const float* __restrict__ input,   // (B, C, padded_H, padded_W)
    float* __restrict__ output,        // (B, C, H, W)
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int search_radius,
    const int patch_radius,
    const float neg_inv_nh    // precomputed: -1.0f / (norm_factor * h_sq)
) {
    const int tm = search_radius + patch_radius;
    const int tile_w = BLOCK_X + 2 * tm;
    const int tile_h = BLOCK_Y + 2 * tm;
    const int tile_size = tile_w * tile_h;
    const int template_size = 2 * patch_radius + 1;

    // D-tile dimensions: covers all patch comparisons for the block
    const int d_w = BLOCK_X + 2 * patch_radius;
    const int d_h = BLOCK_Y + 2 * patch_radius;
    const int d_size = d_w * d_h;

    // Shared memory regions
    extern __shared__ float smem[];
    float* smem_chan = smem;                           // channels × tile_size
    float* smem_d = smem + channels * tile_size;       // d_size
    float* smem_rowfilt = smem_d + d_size;             // d_h × BLOCK_X

    const int ox = blockIdx.x * BLOCK_X + threadIdx.x;
    const int oy = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int b  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_X + tx;

    if (b >= batch_size) return;

    const int padded_w = width + 2 * tm;
    const int padded_h = height + 2 * tm;
    const int batch_stride = channels * padded_h * padded_w;
    const int chan_stride = padded_h * padded_w;
    const int block_start_y = blockIdx.y * BLOCK_Y;
    const int block_start_x = blockIdx.x * BLOCK_X;

    // ── Load all channels into shared memory ONCE ──
    const int total_chan = channels * tile_size;
    for (int idx = tid; idx < total_chan; idx += 256) {
        const int c = idx / tile_size;
        const int pix = idx % tile_size;
        const int sy = pix / tile_w;
        const int sx = pix % tile_w;
        int gy = min(max(block_start_y + sy, 0), padded_h - 1);
        int gx = min(max(block_start_x + sx, 0), padded_w - 1);
        smem_chan[idx] = input[b * batch_stride + c * chan_stride + gy * padded_w + gx];
    }
    __syncthreads();

    // Early exit for out-of-bounds threads (after cooperative smem load)
    if (ox >= width || oy >= height) return;

    float weight_sum = 0.0f;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;

    const int rowfilt_w = BLOCK_X;

    // ── Search loop: all offsets, reads from smem only ──
    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {

            // Step 1: Cooperatively compute D-tile
            // D[y][x] = sum_c (center[c] - shifted[c])^2
            for (int idx = tid; idx < d_size; idx += 256) {
                const int dy_l = idx / d_w;
                const int dx_l = idx % d_w;
                const int cy = dy_l + search_radius;
                const int cx = dx_l + search_radius;
                const int sy = cy + dy;
                const int sx = cx + dx;

                float d = 0.0f;
                for (int c = 0; c < channels; c++) {
                    const float* cs = smem_chan + c * tile_size;
                    float diff = cs[cy * tile_w + cx] - cs[sy * tile_w + sx];
                    d += diff * diff;
                }
                smem_d[idx] = d;
            }
            __syncthreads();

            // Step 2a: Row pass — sum template_size consecutive D values along x
            // Produces d_h × BLOCK_X intermediate values
            const int rowfilt_size = d_h * rowfilt_w;
            for (int idx = tid; idx < rowfilt_size; idx += 256) {
                const int ry = idx / rowfilt_w;
                const int rx = idx % rowfilt_w;
                float sum = 0.0f;
                for (int px = 0; px < template_size; px++) {
                    sum += smem_d[ry * d_w + rx + px];
                }
                smem_rowfilt[idx] = sum;
            }
            __syncthreads();

            // Step 2b: Column pass — each thread sums template_size row-filtered values
            float patch_ssd = 0.0f;
            for (int py = 0; py < template_size; py++) {
                patch_ssd += smem_rowfilt[(ty + py) * rowfilt_w + tx];
            }

            // Step 3: Weight + accumulate (pixel values from channel smem)
            float w = __expf(patch_ssd * neg_inv_nh);
            weight_sum += w;

            int pixel_y = ty + tm + dy;
            int pixel_x = tx + tm + dx;
            r0 += w * smem_chan[0 * tile_size + pixel_y * tile_w + pixel_x];
            r1 += w * smem_chan[1 * tile_size + pixel_y * tile_w + pixel_x];
            r2 += w * smem_chan[2 * tile_size + pixel_y * tile_w + pixel_x];

            __syncthreads();  // before next D-tile write
        }
    }

    // ── Write output ──
    const int obs = channels * height * width;
    const int ocs = height * width;
    float inv_ws = 1.0f / weight_sum;
    output[b * obs + 0 * ocs + oy * width + ox] = r0 * inv_ws;
    output[b * obs + 1 * ocs + oy * width + ox] = r1 * inv_ws;
    output[b * obs + 2 * ocs + oy * width + ox] = r2 * inv_ws;
}


torch::Tensor nlmeans_cuda(
    torch::Tensor input,  // (B, C, H, W) — already padded
    int height,           // original (unpadded) height
    int width,            // original (unpadded) width
    int search_radius,
    int patch_radius,
    float h_val           // h on [0,255] scale
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const float h_scaled = h_val / 255.0f;
    const float h_sq = h_scaled * h_scaled;
    const int template_size = 2 * patch_radius + 1;
    const float norm_factor = template_size * template_size * channels;
    const float neg_inv_nh = -1.0f / (norm_factor * h_sq);

    auto output = torch::zeros({batch_size, channels, height, width}, input.options());

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(
        (width + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y,
        batch_size
    );

    // Shared memory: channel tiles + D tile + row-filtered buffer
    const int tm = search_radius + patch_radius;
    const int tile_w = BLOCK_X + 2 * tm;
    const int tile_h = BLOCK_Y + 2 * tm;
    const int tile_size = tile_w * tile_h;
    const int d_w = BLOCK_X + 2 * patch_radius;
    const int d_h = BLOCK_Y + 2 * patch_radius;
    const int d_size = d_w * d_h;
    const int rowfilt_size = d_h * BLOCK_X;

    const int smem_bytes = (channels * tile_size + d_size + rowfilt_size) * sizeof(float);

    nlmeans_kernel<<<grid, block, smem_bytes>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width,
        search_radius, patch_radius, neg_inv_nh
    );

    return output;
}
