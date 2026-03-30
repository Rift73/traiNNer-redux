"""Pure PyTorch GPU degradation functions — no CPU roundtrips.

Drop-in replacements for apply_per_image-wrapped CPU operations.
All functions take/return BCHW float32 tensors on GPU.

Functions:
- rgb_to_ycbcr_pt / ycbcr_to_rgb_pt: Color space conversion (BT.601/709/2020/240M)
- rgb_to_cmyk_pt / cmyk_to_rgb_pt: CMYK conversion
- channel_shift_pt: Per-channel spatial shift (RGB/YUV/CMYK)
- chroma_subsample_pt: Chroma downsampling + upsampling + optional blur
- quantize_pt: Color quantization
- ordered_dither_pt: Bayer ordered dithering
- dot_diffusion_dither_pt: Knuth's dot diffusion (GPU-parallel error diffusion alternative)
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


# ═══════════════════════════════════════════════════════════════
# Color Space Conversions
# ═══════════════════════════════════════════════════════════════

# YCbCr weights: {standard: (Kr, Kb)}
_YCBCR_WEIGHTS = {
    "601": (0.299, 0.114),
    "709": (0.2126, 0.0722),
    "2020": (0.2627, 0.0593),
    "240": (0.212, 0.087),
}


def _ycbcr_matrix(kr: float, kb: float, device: torch.device) -> Tensor:
    """Build the 3×3 RGB→YCbCr conversion matrix."""
    kg = 1.0 - kr - kb
    # Y  =  Kr*R      + Kg*G      + Kb*B
    # Cb = -Kr/(2*(1-Kb))*R - Kg/(2*(1-Kb))*G + 0.5*B
    # Cr =  0.5*R - Kg/(2*(1-Kr))*G - Kb/(2*(1-Kr))*B
    m = torch.tensor(
        [
            [kr, kg, kb],
            [-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), 0.5],
            [0.5, -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))],
        ],
        dtype=torch.float32,
        device=device,
    )
    return m


def rgb_to_ycbcr_pt(tensor: Tensor, standard: str = "709") -> Tensor:
    """RGB [0,1] → YCbCr. Y in [0,1], Cb/Cr in [-0.5, 0.5].

    Args:
        tensor: BCHW float32 tensor, C=3 (RGB).
        standard: "601", "709", "2020", or "240".

    Returns:
        BCHW float32 tensor, C=3 (Y, Cb, Cr).
    """
    kr, kb = _YCBCR_WEIGHTS[standard]
    m = _ycbcr_matrix(kr, kb, tensor.device)  # (3, 3)
    # BCHW → B,3,H*W → matmul → B,3,H*W → BCHW
    b, _c, h, w = tensor.shape
    flat = tensor.reshape(b, 3, h * w)  # (B, 3, N)
    ycbcr = torch.matmul(m, flat)  # (3, 3) × (B, 3, N) → (B, 3, N) via broadcast
    return ycbcr.reshape(b, 3, h, w)


def ycbcr_to_rgb_pt(tensor: Tensor, standard: str = "709") -> Tensor:
    """YCbCr → RGB [0,1]. Inverse of rgb_to_ycbcr_pt.

    Args:
        tensor: BCHW float32 tensor, C=3 (Y, Cb, Cr).
        standard: "601", "709", "2020", or "240".

    Returns:
        BCHW float32 tensor, C=3 (RGB), clamped to [0, 1].
    """
    kr, kb = _YCBCR_WEIGHTS[standard]
    m = _ycbcr_matrix(kr, kb, tensor.device)
    m_inv = torch.linalg.inv(m)  # (3, 3)
    b, _c, h, w = tensor.shape
    flat = tensor.reshape(b, 3, h * w)
    rgb = torch.matmul(m_inv, flat)
    return rgb.reshape(b, 3, h, w).clamp(0, 1)


def rgb_to_cmyk_pt(tensor: Tensor) -> Tensor:
    """RGB [0,1] → CMYK [0,1]. Returns B,4,H,W tensor.

    Standard formula: K = 1 - max(R,G,B), C = (1-R-K)/(1-K), etc.
    """
    r, g, b_ = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
    k = 1.0 - torch.max(tensor, dim=1, keepdim=True).values
    denom = (1.0 - k).clamp(min=1e-6)  # avoid division by zero
    c = (1.0 - r - k) / denom
    m = (1.0 - g - k) / denom
    y = (1.0 - b_ - k) / denom
    return torch.cat([c, m, y, k], dim=1)


def cmyk_to_rgb_pt(tensor: Tensor) -> Tensor:
    """CMYK [0,1] → RGB [0,1]. Input is B,4,H,W tensor."""
    c, m, y, k = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3], tensor[:, 3:4]
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b_ = (1.0 - y) * (1.0 - k)
    return torch.cat([r, g, b_], dim=1).clamp(0, 1)


# ═══════════════════════════════════════════════════════════════
# Channel Shift
# ═══════════════════════════════════════════════════════════════


def _shift_channel(
    channel: Tensor, dx: int, dy: int, fill_value: float = 1.0
) -> Tensor:
    """Shift a single channel (B, 1, H, W) by (dx, dy) pixels.

    Positive dx shifts content RIGHT (new pixels appear on left).
    Positive dy shifts content DOWN (new pixels appear on top).
    Empty space filled with fill_value (matching cv2.warpAffine BORDER_CONSTANT).
    """
    if dx == 0 and dy == 0:
        return channel

    _, _, h, w = channel.shape
    result = torch.full_like(channel, fill_value)

    # Source and destination slices
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)

    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)

    if src_y_end > src_y_start and src_x_end > src_x_start:
        result[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            channel[:, :, src_y_start:src_y_end, src_x_start:src_x_end]

    return result


def channel_shift_pt(
    tensor: Tensor,
    shift_type: str,
    amounts: list[list[list[int]]],
    percent: bool = False,
) -> Tensor:
    """Apply per-channel spatial shift on GPU.

    Args:
        tensor: BCHW float32 [0,1] RGB tensor on GPU.
        shift_type: "rgb", "yuv", or "cmyk".
        amounts: Per-channel shift ranges. Each entry is [[x_lo, x_hi], [y_lo, y_hi]].
        percent: If True, amounts are percentages of image dimensions.

    Returns:
        BCHW float32 shifted tensor.
    """
    import numpy as np

    _, _, h, w = tensor.shape

    def _sample_int(lo, hi):
        """Sample integer from [lo, hi] inclusive. Handles lo==hi."""
        if lo == hi:
            return lo
        if lo > hi:
            lo, hi = hi, lo
        return np.random.randint(lo, hi + 1)

    def sample_shift(amount_range):
        ax_range, ay_range = amount_range
        if percent:
            ax = 0 if ax_range == [0, 0] else int(h * np.random.uniform(*ax_range) / 100)
            ay = 0 if ay_range == [0, 0] else int(w * np.random.uniform(*ay_range) / 100)
        else:
            ax = 0 if ax_range == [0, 0] else _sample_int(ax_range[0], ax_range[1])
            ay = 0 if ay_range == [0, 0] else _sample_int(ay_range[0], ay_range[1])
        return ax, ay

    if shift_type == "rgb":
        result = tensor.clone()
        for c in range(3):
            dx, dy = sample_shift(amounts[c])
            result[:, c : c + 1] = _shift_channel(tensor[:, c : c + 1], dx, dy, fill_value=1.0)
        return result

    elif shift_type == "yuv":
        ycbcr = rgb_to_ycbcr_pt(tensor, standard="2020")
        for c in range(3):
            dx, dy = sample_shift(amounts[c])
            ycbcr[:, c : c + 1] = _shift_channel(ycbcr[:, c : c + 1], dx, dy, fill_value=1.0)
        return ycbcr_to_rgb_pt(ycbcr, standard="2020")

    elif shift_type == "cmyk":
        cmyk = rgb_to_cmyk_pt(tensor)
        for c in range(4):
            dx, dy = sample_shift(amounts[c])
            cmyk[:, c : c + 1] = _shift_channel(cmyk[:, c : c + 1], dx, dy, fill_value=0.0)
        return cmyk_to_rgb_pt(cmyk)

    return tensor


# ═══════════════════════════════════════════════════════════════
# Chroma Subsampling
# ═══════════════════════════════════════════════════════════════

# Subsampling format → [Y_v_scale, Cb_v_scale, Cr_h_scale]
_SUBSAMPLING_MAP = {
    "4:4:4": (1.0, 1.0),    # no subsampling
    "4:2:2": (1.0, 0.5),    # chroma half-width
    "4:2:0": (0.5, 0.5),    # chroma half both
    "4:1:1": (1.0, 0.25),   # chroma quarter-width
    "4:1:0": (0.5, 0.25),
    "4:4:0": (0.5, 1.0),
    "4:2:1": (0.5, 0.5),
    "4:1:2": (1.0, 0.25),
    "4:1:3": (0.75, 0.25),
}

# F.interpolate mode mapping (closest equivalents for GPU)
_INTERPOLATE_MODE_MAP = {
    "nearest": "nearest",
    "box": "nearest",        # closest GPU equivalent
    "hermite": "bilinear",   # closest GPU equivalent
    "linear": "bilinear",
    "lagrange": "bilinear",
    "cubic_catrom": "bicubic",
    "cubic_mitchell": "bicubic",
    "cubic_bspline": "bicubic",
    "lanczos": "bicubic",    # closest GPU equivalent
    "gauss": "bilinear",
}


def _make_gaussian_kernel(sigma: float, device: torch.device) -> Tensor:
    """Create a Gaussian blur kernel for F.conv2d."""
    radius = int(3.0 * sigma + 0.5)
    if radius < 1:
        radius = 1
    size = 2 * radius + 1
    x = torch.arange(size, dtype=torch.float32, device=device) - radius
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)


def chroma_subsample_pt(
    tensor: Tensor,
    down_mode: str = "bilinear",
    up_mode: str = "bilinear",
    format_str: str = "4:2:0",
    blur_sigma: float | None = None,
    ycbcr_type: str = "709",
) -> Tensor:
    """Apply chroma subsampling on GPU.

    Args:
        tensor: BCHW float32 [0,1] RGB tensor.
        down_mode: Downsample interpolation mode name.
        up_mode: Upsample interpolation mode name.
        format_str: Subsampling format (e.g. "4:2:0").
        blur_sigma: Optional Gaussian blur sigma for chroma channels.
        ycbcr_type: YCbCr standard ("601", "709", "2020", "240").

    Returns:
        BCHW float32 [0,1] RGB tensor.
    """
    if format_str == "4:4:4":
        return tensor

    v_scale, h_scale = _SUBSAMPLING_MAP[format_str]
    d_mode = _INTERPOLATE_MODE_MAP.get(down_mode, "bilinear")
    u_mode = _INTERPOLATE_MODE_MAP.get(up_mode, "bilinear")
    align = d_mode != "nearest"

    # RGB → YCbCr
    ycbcr = rgb_to_ycbcr_pt(tensor, standard=ycbcr_type)
    b, c, h, w = ycbcr.shape

    # Extract chroma channels (Cb, Cr)
    chroma = ycbcr[:, 1:3]  # (B, 2, H, W)

    # Downsample chroma
    down_h = max(1, int(h * v_scale))
    down_w = max(1, int(w * h_scale))
    chroma_down = F.interpolate(
        chroma, size=(down_h, down_w), mode=d_mode,
        align_corners=align if d_mode != "nearest" else None,
    )

    # Upsample back
    chroma_up = F.interpolate(
        chroma_down, size=(h, w), mode=u_mode,
        align_corners=align if u_mode != "nearest" else None,
    )

    # Optional blur on chroma
    if blur_sigma is not None and blur_sigma > 0:
        kernel = _make_gaussian_kernel(blur_sigma, tensor.device)
        pad_size = kernel.shape[-1] // 2
        # Apply blur to each chroma channel
        cb = F.conv2d(
            F.pad(chroma_up[:, 0:1], [pad_size] * 4, mode="reflect"),
            kernel,
        )
        cr = F.conv2d(
            F.pad(chroma_up[:, 1:2], [pad_size] * 4, mode="reflect"),
            kernel,
        )
        chroma_up = torch.cat([cb, cr], dim=1)

    # Replace chroma in YCbCr
    ycbcr = torch.cat([ycbcr[:, 0:1], chroma_up], dim=1)

    # YCbCr → RGB
    return ycbcr_to_rgb_pt(ycbcr, standard=ycbcr_type)


# ═══════════════════════════════════════════════════════════════
# Dithering
# ═══════════════════════════════════════════════════════════════


def quantize_pt(tensor: Tensor, levels: int) -> Tensor:
    """Uniform quantization on GPU.

    Args:
        tensor: BCHW float32 [0,1] tensor.
        levels: Number of quantization levels per channel (e.g., 8 = 3-bit).

    Returns:
        BCHW float32 [0,1] quantized tensor.
    """
    n = levels - 1
    return torch.round(tensor * n) / n


def _bayer_matrix(n: int, device: torch.device) -> Tensor:
    """Generate an n×n Bayer ordered dither threshold matrix.

    Recursive construction: M(2n) = [[4*M(n), 4*M(n)+2], [4*M(n)+3, 4*M(n)+1]] / (4n²)
    """
    if n == 1:
        return torch.zeros(1, 1, dtype=torch.float32, device=device)

    # Build recursively
    if n == 2:
        m = torch.tensor([[0, 2], [3, 1]], dtype=torch.float32, device=device)
        return m / 4.0

    half = n // 2
    m_half = _bayer_matrix(half, device) * (half * half)  # un-normalize
    m = torch.zeros(n, n, dtype=torch.float32, device=device)
    m[:half, :half] = 4 * m_half
    m[:half, half:] = 4 * m_half + 2
    m[half:, :half] = 4 * m_half + 3
    m[half:, half:] = 4 * m_half + 1
    return m / (n * n)


def ordered_dither_pt(tensor: Tensor, levels: int, map_size: int = 4) -> Tensor:
    """Ordered (Bayer) dithering on GPU.

    Args:
        tensor: BCHW float32 [0,1] tensor.
        levels: Number of quantization levels per channel.
        map_size: Bayer matrix size (must be power of 2).

    Returns:
        BCHW float32 [0,1] dithered tensor.
    """
    # Round map_size to nearest power of 2
    ms = max(2, 1 << (map_size - 1).bit_length())

    bayer = _bayer_matrix(ms, tensor.device)  # (ms, ms) in [0, 1)
    _, _, h, w = tensor.shape

    # Tile Bayer matrix across image
    # Use modular indexing to tile without allocating full-size tensor
    y_idx = torch.arange(h, device=tensor.device) % ms
    x_idx = torch.arange(w, device=tensor.device) % ms
    threshold = bayer[y_idx[:, None], x_idx[None, :]]  # (H, W)
    threshold = threshold.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) — broadcasts over B, C

    # Dither: add threshold offset before quantization
    n = levels - 1
    dithered = torch.floor(tensor * n + threshold) / n
    return dithered.clamp(0, 1)


def dot_diffusion_dither_pt(tensor: Tensor, levels: int) -> Tensor:
    """Dot diffusion dithering on GPU — Knuth's parallel error diffusion.

    Uses an 8×8 class matrix. All pixels of the same class are processed
    simultaneously (fully parallel within each class). 64 classes = 64 serial
    steps, but each step processes ~1/64 of all pixels in parallel.

    Error is distributed to 8-connected neighbors via torch.roll (vectorized,
    no per-pixel Python loops).

    Args:
        tensor: BCHW float32 [0,1] tensor.
        levels: Number of quantization levels per channel.

    Returns:
        BCHW float32 [0,1] dithered tensor.
    """
    # Knuth's 8×8 class matrix (processing order)
    class_matrix = torch.tensor(
        [
            [34, 48, 40, 32, 29, 15, 23, 31],
            [42, 58, 56, 53, 21, 5, 7, 10],
            [50, 62, 61, 45, 13, 1, 2, 18],
            [38, 46, 54, 37, 25, 17, 9, 26],
            [28, 14, 22, 30, 35, 49, 41, 33],
            [20, 4, 6, 11, 43, 59, 57, 52],
            [12, 0, 3, 19, 51, 63, 60, 44],
            [24, 16, 8, 27, 39, 47, 55, 36],
        ],
        dtype=torch.int64,
        device=tensor.device,
    )

    # 8-connected neighbor offsets and equal weights
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    weight = 1.0 / 8.0  # equal weight to all 8 neighbors

    _b, _c, h, w = tensor.shape
    n = levels - 1

    # Tile class matrix across image: (H, W)
    y_idx = torch.arange(h, device=tensor.device) % 8
    x_idx = torch.arange(w, device=tensor.device) % 8
    class_map = class_matrix[y_idx[:, None], x_idx[None, :]]

    result = tensor.clone()

    # Process each class in order
    for cls in range(64):
        # Mask for pixels of this class: (1, 1, H, W) for broadcasting
        mask = (class_map == cls).unsqueeze(0).unsqueeze(0).float()

        # Save values before quantization (only at masked positions)
        old_val = result * mask

        # Quantize
        quantized = torch.round(result * n) / n

        # Apply quantization only to this class's pixels
        result = result * (1.0 - mask) + quantized * mask

        # Error at this class's pixels (zero elsewhere)
        error = (old_val - result * mask)  # error only where mask=1

        # Distribute error to 8 neighbors via torch.roll
        # Only add to pixels in higher classes (not yet processed)
        higher_mask = (class_map > cls).unsqueeze(0).unsqueeze(0).float()

        for dy, dx in neighbors:
            # Roll error to neighbor position
            shifted_error = torch.roll(error, shifts=(dy, dx), dims=(2, 3))
            # Add weighted error only to higher-class pixels
            result = result + shifted_error * higher_mask * weight

    return result.clamp(0, 1)
