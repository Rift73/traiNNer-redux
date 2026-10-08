"""Exact CUDA error-diffusion and Riemersma dithering for BCHW batches.

Bit-identical, image by image, to chainner_ext's error_diffusion_dither and
riemersma_dither with UniformQuantization. JIT-compiles on first use (torch's
build cache makes later processes reuse the build); dither_cuda_available()
reports whether the extension can be used, so callers keep the per-image CPU
path as the fallback.

Usage:
    from traiNNer.data.dither_cuda import dither_cuda_available, error_diffusion_dither_cuda
    if dither_cuda_available():
        out = error_diffusion_dither_cuda(x, levels=4, algorithm="floydsteinberg")
"""

from __future__ import annotations

import logging
from functools import cache, lru_cache
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor

logger = logging.getLogger("traiNNer")

# chainner_ext.DiffusionAlgorithm order.
DIFFUSION_ALGORITHMS: dict[str, int] = {
    "floydsteinberg": 0,
    "jarvisjudiceninke": 1,
    "stucki": 2,
    "atkinson": 3,
    "burkes": 4,
    "sierra": 5,
    "tworowsierra": 6,
    "sierralite": 7,
}

_csrc_dir = Path(__file__).resolve().parent.parent / "csrc"


@cache
def _extension() -> ModuleType | None:
    if not torch.cuda.is_available():
        logger.info("CUDA dithering unavailable (no CUDA device); using the CPU path.")
        return None
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="dither_cuda_ext",
            sources=[
                str(_csrc_dir / "dither.cpp"),
                str(_csrc_dir / "dither_kernel.cu"),
            ],
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            "CUDA dithering extension unavailable (%s); using the CPU path.", exc
        )
        return None


def dither_cuda_available() -> bool:
    """Build the extension on first call; False when it cannot be used."""
    return _extension() is not None


def _require_extension() -> ModuleType:
    ext = _extension()
    if ext is None:
        raise RuntimeError("CUDA dithering extension is unavailable")
    return ext


@lru_cache(maxsize=8)
def _riemersma_order(
    height: int, width: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """The traversal order and its inverse permutation, built once per size."""
    order, inverse = _require_extension().riemersma_order(height, width)
    return order.to(device), inverse.to(device)


def _levels(levels: int | Tensor, x: Tensor) -> Tensor:
    batch = x.shape[0]
    if isinstance(levels, Tensor):
        per_sample = levels.to(device=x.device, dtype=torch.int64).reshape(batch)
        if bool((per_sample < 2).any()):
            raise ValueError("Every quantization level count must be at least 2.")
        return per_sample.contiguous()
    if levels < 2:
        raise ValueError("The quantization level count must be at least 2.")
    return torch.full((batch,), levels, dtype=torch.int64, device=x.device)


def error_diffusion_dither_cuda(
    x: Tensor, levels: int | Tensor, algorithm: str
) -> Tensor:
    """Error-diffusion dither every image of a BCHW CUDA batch.

    Args:
        x: BCHW CUDA tensor with 1, 3 or 4 channels, values nominally in [0, 1].
        levels: Quantization levels per channel, one int or an int tensor of B.
        algorithm: A key of DIFFUSION_ALGORITHMS.

    Returns:
        New dithered tensor, same shape and dtype as x.
    """
    src = x.detach().float().contiguous()
    out = _require_extension().error_diffusion(
        src, _levels(levels, src), DIFFUSION_ALGORITHMS[algorithm]
    )
    return out.to(x.dtype)


def riemersma_dither_cuda(
    x: Tensor, levels: int | Tensor, history: int, decay_ratio: float
) -> Tensor:
    """Riemersma dither every image of a BCHW CUDA batch.

    Args:
        x: BCHW CUDA tensor with 1, 3 or 4 channels, values nominally in [0, 1].
        levels: Quantization levels per channel, one int or an int tensor of B.
        history: History length (at least 2).
        decay_ratio: Weight of the oldest history entry, in (0, 1).

    Returns:
        New dithered tensor, same shape and dtype as x.
    """
    src = x.detach().float().contiguous()
    order, inverse = _riemersma_order(src.shape[2], src.shape[3], src.device)
    out = _require_extension().riemersma(
        src, _levels(levels, src), order, inverse, history, decay_ratio
    )
    return out.to(x.dtype)
