"""Exact CUDA error-diffusion, Riemersma and palette dithering for BCHW batches.

Bit-identical, image by image, to chainner_ext's error_diffusion_dither,
riemersma_dither and quantize with UniformQuantization and PaletteQuantization.
JIT-compiles on first use (torch's build cache makes later processes reuse the
build); dither_cuda_available() reports whether the extension can be used, so
callers keep the per-image CPU path as the fallback.

Usage:
    from traiNNer.data.dither_cuda import dither_cuda_available, error_diffusion_dither_cuda
    if dither_cuda_available():
        out = error_diffusion_dither_cuda(x, levels=4, algorithm="floydsteinberg")
        out = palette_dither_cuda(x, "floydsteinberg", num_colors=64, history=12, decay_ratio=0.5)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache, lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import Tensor

from traiNNer.data.otf_degradations import _extract_palette

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

# The dithering types apply_dithering_palette maps onto the palette kernels
# ("quantize" and "order" both take the palette's quantize).
PALETTE_DITHERING_TYPES = frozenset(
    {*DIFFUSION_ALGORITHMS, "riemersma", "quantize", "order"}
)

# Threads for the batch's median cuts (Pillow releases the GIL in quantize).
_PALETTE_WORKERS = 16

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


def _palette_order(palette: np.ndarray) -> np.ndarray:
    """A palette's colors in chainner_ext's PaletteQuantization order.

    The unique colors (by bits) sorted by the key of chaiNNer-rs's
    extract_unique (the color for one channel, else r*r*0.2126 + g*g*0.7152 +
    b*b*0.0722 in float32) in f32::total_cmp order. Equal keys keep their first
    occurrence first; upstream leaves them in hash-set order, which only an
    exact distance tie between two equal-key colors could expose.
    """
    channels = palette.shape[-1]
    colors = np.ascontiguousarray(palette, dtype=np.float32).reshape(-1, channels)
    _, first = np.unique(colors.view(np.uint32), axis=0, return_index=True)
    colors = colors[np.sort(first)]
    if channels == 1:
        key = colors[:, 0]
    else:
        squares = colors * colors
        key = (
            squares[:, 0] * np.float32(0.2126)
            + squares[:, 1] * np.float32(0.7152)
            + squares[:, 2] * np.float32(0.0722)
        )
    bits = np.ascontiguousarray(key).view(np.uint32)
    flip = np.where(bits >> 31 != 0, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    return colors[np.argsort(bits ^ flip, kind="stable")]


def palette_quantization_cuda(
    palettes: Sequence[np.ndarray], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Upload one palette per image as the palette kernels take them.

    Args:
        palettes: One (P, C) or (1, P, C) float32 array per image, P at most
            256, C 1 or 3, as _extract_palette returns them.
        device: The CUDA device of the batch.

    Returns:
        (N, Pmax, C) float32 palettes in PaletteQuantization's order, zero
        padded, and the int64 color count of each.
    """
    ordered = [_palette_order(palette) for palette in palettes]
    counts = [len(colors) for colors in ordered]
    padded = np.zeros(
        (len(ordered), max(counts), ordered[0].shape[1]), dtype=np.float32
    )
    for i, colors in enumerate(ordered):
        padded[i, : counts[i]] = colors
    return (
        torch.from_numpy(padded).to(device),
        torch.tensor(counts, dtype=torch.int64).to(device),
    )


def palette_quantize_cuda(x: Tensor, palettes: Tensor, counts: Tensor) -> Tensor:
    """Map every pixel of a BCHW CUDA batch to its image's nearest palette color.

    Args:
        x: BCHW CUDA tensor with 1 or 3 channels.
        palettes: (N, Pmax, C) float32 palettes from palette_quantization_cuda.
        counts: Their int64 color counts.

    Returns:
        New tensor, same shape and dtype as x.
    """
    src = x.detach().float().contiguous()
    return _require_extension().palette_quantize(src, palettes, counts).to(x.dtype)


def palette_error_diffusion_dither_cuda(
    x: Tensor, palettes: Tensor, counts: Tensor, algorithm: str
) -> Tensor:
    """Error-diffusion dither every image of a BCHW CUDA batch to its palette.

    Args:
        x: BCHW CUDA tensor with 1 or 3 channels, values nominally in [0, 1].
        palettes: (N, Pmax, C) float32 palettes from palette_quantization_cuda.
        counts: Their int64 color counts.
        algorithm: A key of DIFFUSION_ALGORITHMS.

    Returns:
        New dithered tensor, same shape and dtype as x.
    """
    src = x.detach().float().contiguous()
    out = _require_extension().palette_error_diffusion(
        src, palettes, counts, DIFFUSION_ALGORITHMS[algorithm]
    )
    return out.to(x.dtype)


def palette_riemersma_dither_cuda(
    x: Tensor, palettes: Tensor, counts: Tensor, history: int, decay_ratio: float
) -> Tensor:
    """Riemersma dither every image of a BCHW CUDA batch to its palette.

    Args:
        x: BCHW CUDA tensor with 1 or 3 channels, values nominally in [0, 1].
        palettes: (N, Pmax, C) float32 palettes from palette_quantization_cuda.
        counts: Their int64 color counts.
        history: History length (at least 2).
        decay_ratio: Weight of the oldest history entry, in (0, 1).

    Returns:
        New dithered tensor, same shape and dtype as x.
    """
    src = x.detach().float().contiguous()
    order, inverse = _riemersma_order(src.shape[2], src.shape[3], src.device)
    out = _require_extension().palette_riemersma(
        src, palettes, counts, order, inverse, history, decay_ratio
    )
    return out.to(x.dtype)


@cache
def _palette_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=min(_PALETTE_WORKERS, os.cpu_count() or 1))


def _dither_to_palettes(
    x: Tensor,
    palettes: Sequence[np.ndarray],
    dithering_type: str,
    history: int,
    decay_ratio: float,
) -> Tensor:
    uploaded, counts = palette_quantization_cuda(palettes, x.device)
    if dithering_type in DIFFUSION_ALGORITHMS:
        return palette_error_diffusion_dither_cuda(x, uploaded, counts, dithering_type)
    if dithering_type == "riemersma":
        return palette_riemersma_dither_cuda(x, uploaded, counts, history, decay_ratio)
    return palette_quantize_cuda(x, uploaded, counts)


def palette_dither_cuda(
    x: Tensor, dithering_type: str, num_colors: int, history: int, decay_ratio: float
) -> Tensor:
    """apply_dithering_palette over a BCHW CUDA batch.

    Pillow's median cut runs per image on the CPU (threads over the downloaded
    batch), exactly as the CPU path's, and the dithering on the GPU. An image
    without a usable palette comes back unchanged, as on the CPU path; with 4
    channels the palette covers the color channels and alpha passes through.

    Args:
        x: BCHW CUDA tensor with 1, 3 or 4 channels, values nominally in [0, 1].
        dithering_type: A member of PALETTE_DITHERING_TYPES.
        num_colors: Number of palette colors (2-256).
        history: History length for Riemersma dithering.
        decay_ratio: Decay ratio for Riemersma dithering.

    Returns:
        New tensor, same shape and dtype as x.
    """
    if x.shape[1] == 4:
        color = palette_dither_cuda(
            x[:, :3], dithering_type, num_colors, history, decay_ratio
        )
        return torch.cat((color, x[:, 3:]), dim=1)
    images = x.detach().float().cpu().numpy()
    palettes = list(
        _palette_pool().map(
            lambda image: _extract_palette(image.transpose(1, 2, 0), num_colors), images
        )
    )
    ready = [i for i, palette in enumerate(palettes) if palette is not None]
    if len(ready) == len(palettes):
        return _dither_to_palettes(x, palettes, dithering_type, history, decay_ratio)
    out = x.clone()
    if ready:
        index = torch.tensor(ready, device=x.device)
        out[index] = _dither_to_palettes(
            x[index], [palettes[i] for i in ready], dithering_type, history, decay_ratio
        )
    return out
