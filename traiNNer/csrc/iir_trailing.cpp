// PyTorch C++ binding for IIR trailing CUDA kernel.

#include <torch/extension.h>

void iir_trailing_forward_cuda(
    float* output, const float* input,
    float alpha, int num_rows, int W
);

torch::Tensor iir_trailing_forward(torch::Tensor input, float alpha) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const int ndim = input.dim();
    const int W = input.size(ndim - 1);
    const int num_rows = input.numel() / W;

    auto output = input.clone();

    if (W <= 1 || alpha >= 1.0f) return output;

    iir_trailing_forward_cuda(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        alpha, num_rows, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("iir_trailing_forward", &iir_trailing_forward,
          "Causal 1-pole IIR trailing filter (CUDA)");
}
