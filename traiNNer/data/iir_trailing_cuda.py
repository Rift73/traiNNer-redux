"""Lazy-loading CUDA IIR trailing filter (tape trailing effect).

JIT-compiles on first use. Follows the same pattern as nlmeans_cuda.py.
"""

from pathlib import Path

from torch import Tensor
from torch.utils.cpp_extension import load

_csrc_dir = str(Path(__file__).resolve().parent.parent / "csrc")

_iir_ext = load(
    name="iir_trailing_cuda_ext",
    sources=[
        f"{_csrc_dir}/iir_trailing.cpp",
        f"{_csrc_dir}/iir_trailing_kernel.cu",
    ],
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    verbose=False,
)


def iir_trailing_cuda(signal: Tensor, strength: float) -> Tensor:
    """Causal 1-pole IIR trailing filter using CUDA kernel.

    Args:
        signal: Any shape float32 CUDA tensor. IIR applied along last dim.
        strength: 0-1, maps to alpha = 1 - strength * 0.70.

    Returns:
        Filtered tensor (same shape).
    """
    if strength <= 0:
        return signal

    alpha = 1.0 - min(max(strength, 0.0), 1.0) * 0.70

    return _iir_ext.iir_trailing_forward(signal.contiguous(), alpha)
