// Causal 1-pole IIR filter (tape trailing) — one thread per row.
// y[i] = alpha * x[i] + (1 - alpha) * y[i-1], applied along last dimension.

#include <cuda_runtime.h>

__global__ void iir_trailing_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    const float alpha,
    const float beta,
    const int num_rows,
    const int W
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const float* in_row = inp + row * W;
    float* out_row = out + row * W;

    float prev = in_row[0];
    out_row[0] = prev;

    for (int i = 1; i < W; i++) {
        prev = alpha * in_row[i] + beta * prev;
        out_row[i] = prev;
    }
}

void iir_trailing_forward_cuda(
    float* output, const float* input,
    float alpha, int num_rows, int W
) {
    const float beta = 1.0f - alpha;
    const int threads = 256;
    const int blocks = (num_rows + threads - 1) / threads;

    iir_trailing_kernel<<<blocks, threads>>>(
        output, input, alpha, beta, num_rows, W
    );
}
