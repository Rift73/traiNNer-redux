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
- composite_rainbow_pt: NTSC/PAL composite video chroma dot crawl
- lowpass_filter_pt: Butterworth frequency-domain lowpass (anime mastering artifact)
- interlace_pt: Interlaced video combing artifact
- overshoot_pt: Edge ringing from re-sharpening (warp sharp)
- temporal_ghosting_pt: Residual frame blending (video ghosting)
- scanline_pt: CRT scanline darkening
- detail_mask_neo_pt: Edge/detail mask (for lowpass detail preservation)
- bicubic_descale: Inverse bicubic upscale (for NTSC resolution matching)
- ntsc_composite_pt: Full NTSC composite encode/effects/decode simulation
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import functional as F

logger = logging.getLogger(__name__)

try:
    from traiNNer.data.iir_trailing_cuda import iir_trailing_cuda

    _HAS_IIR_CUDA = True
except Exception:
    iir_trailing_cuda = None  # type: ignore[assignment]
    _HAS_IIR_CUDA = False


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


# ═══════════════════════════════════════════════════════════════
# Composite Rainbow (NTSC chroma dot crawl)
# ═══════════════════════════════════════════════════════════════

# RGB ↔ YIQ matrices
_YIQ_FWD = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [0.5959, -0.2746, -0.3213],
        [0.2115, -0.5227, 0.3112],
    ],
    dtype=torch.float32,
)
_YIQ_INV = torch.tensor(
    [
        [1.0, 0.956, 0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106, 1.703],
    ],
    dtype=torch.float32,
)
_yiq_cache: dict[torch.device, tuple[Tensor, Tensor]] = {}


def _get_yiq_matrices(device: torch.device) -> tuple[Tensor, Tensor]:
    """Get cached RGB↔YIQ matrices for the given device."""
    if device not in _yiq_cache:
        _yiq_cache[device] = (_YIQ_FWD.to(device), _YIQ_INV.to(device))
    return _yiq_cache[device]


def composite_rainbow_pt(
    tensor: Tensor,
    subcarrier_freq: float = 0.25,
    chroma_bandwidth: float = 0.08,
    intensity: float = 1.0,
    phase_alternation: bool = True,
    phase_offset: float = 0.0,
) -> Tensor:
    """Simulate NTSC composite video rainbow artifact (chroma dot crawl).

    Models the encode→decode path of composite video where luma and chroma
    share one signal via a color subcarrier (~3.58 MHz NTSC). Imperfect
    comb-filter separation causes high-frequency luma to leak into chroma,
    creating rainbow-colored fringes along sharp edges.

    Args:
        tensor: BCHW float32 [0,1] RGB tensor.
        subcarrier_freq: Subcarrier frequency in cycles/pixel (0.15–0.35 typical).
        chroma_bandwidth: Chroma demod bandwidth in cycles/pixel.
        intensity: Blend factor (0 = no effect, 1 = full composite decode).
        phase_alternation: If True, subcarrier phase alternates by π each scanline.
        phase_offset: Initial phase offset in radians.

    Returns:
        BCHW float32 [0,1] tensor with composite rainbow artifacts.
    """
    b, _c, h, w = tensor.shape
    device = tensor.device

    m_fwd, m_inv = _get_yiq_matrices(device)
    flat = tensor.reshape(b, 3, h * w)
    yiq = torch.matmul(m_fwd, flat).reshape(b, 3, h, w)
    y = yiq[:, 0:1]
    i_ch = yiq[:, 1:2]
    q_ch = yiq[:, 2:3]

    omega = 2.0 * 3.141592653589793 * subcarrier_freq
    x = torch.arange(w, device=device, dtype=torch.float32)
    if phase_alternation:
        line_phase = (
            torch.arange(h, device=device, dtype=torch.float32) * 3.141592653589793
        )
    else:
        line_phase = torch.zeros(h, device=device, dtype=torch.float32)

    phase = omega * x.unsqueeze(0) + line_phase.unsqueeze(1) + phase_offset
    carrier_cos = torch.cos(phase).unsqueeze(0).unsqueeze(0)
    carrier_sin = torch.sin(phase).unsqueeze(0).unsqueeze(0)

    composite = y + i_ch * carrier_cos + q_ch * carrier_sin

    sigma = 1.0 / (2.0 * 3.141592653589793 * max(chroma_bandwidth, 0.01))
    radius = min(int(3.0 * sigma + 0.5), w // 2)
    radius = max(radius, 1)
    ksize = 2 * radius + 1
    kx = torch.arange(ksize, device=device, dtype=torch.float32) - radius
    kernel = torch.exp(-0.5 * (kx / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.reshape(1, 1, 1, ksize)

    i_demod = composite * (2.0 * carrier_cos)
    q_demod = composite * (2.0 * carrier_sin)

    iq_demod = torch.cat([i_demod, q_demod], dim=0)
    iq_filtered = F.conv2d(
        F.pad(iq_demod, [radius, radius, 0, 0], mode="reflect"),
        kernel,
    )
    i_decoded = iq_filtered[:b]
    q_decoded = iq_filtered[b:]

    yiq_decoded = torch.cat([y, i_decoded, q_decoded], dim=1)
    flat_decoded = yiq_decoded.reshape(b, 3, h * w)
    rgb_decoded = torch.matmul(m_inv, flat_decoded).reshape(b, 3, h, w)

    if intensity < 1.0:
        rgb_decoded = tensor + intensity * (rgb_decoded - tensor)

    return rgb_decoded.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════
# Lowpass Filter (frequency domain — anime mastering artifact)
# ═══════════════════════════════════════════════════════════════


def lowpass_filter_pt(
    tensor: Tensor,
    cutoff: float = 0.5,
    order: int = 2,
) -> Tensor:
    """Apply Butterworth lowpass filter in the frequency domain.

    Simulates production/mastering lowpass filtering common in anime sources
    and broadcast video. Produces bandwidth-limited detail loss with ringing
    (Gibbs phenomenon) at edges when the filter order is high.

    Args:
        tensor: BCHW float32 [0,1] tensor.
        cutoff: Cutoff frequency as fraction of Nyquist (0–1).
        order: Butterworth filter order.

    Returns:
        BCHW float32 [0,1] filtered tensor.
    """
    _b, _c, h, w = tensor.shape
    device = tensor.device

    cutoff_freq = max(cutoff * 0.5, 1e-6)

    freq_y = torch.fft.fftfreq(h, device=device, dtype=torch.float32)
    freq_x = torch.fft.rfftfreq(w, device=device, dtype=torch.float32)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing="ij")
    dist = torch.sqrt(fy * fy + fx * fx)

    butterworth = 1.0 / (1.0 + (dist / cutoff_freq) ** (2 * order))
    butterworth = butterworth.unsqueeze(0).unsqueeze(0)

    spectrum = torch.fft.rfft2(tensor)
    result = torch.fft.irfft2(spectrum * butterworth, s=(h, w))

    return result.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════
# Interlacing / Combing Artifact
# ═══════════════════════════════════════════════════════════════


def interlace_pt(
    tensor: Tensor,
    field_shift: int = 1,
    dominant_field: str = "top",
) -> Tensor:
    """Simulate interlaced video combing artifact.

    Creates the characteristic horizontal combing pattern seen in poorly
    deinterlaced 480i/576i video. Even scanlines from one field, odd from
    another — with the second field spatially shifted.

    Args:
        tensor: BCHW float32 [0,1] tensor.
        field_shift: Horizontal pixel displacement between fields.
        dominant_field: "top" or "bottom".

    Returns:
        BCHW float32 [0,1] tensor with combing artifacts.
    """
    if field_shift == 0:
        return tensor

    result = tensor.clone()

    if dominant_field == "top":
        shifted = torch.roll(tensor, shifts=field_shift, dims=3)
        result[:, :, 1::2, :] = shifted[:, :, 1::2, :]
    else:
        shifted = torch.roll(tensor, shifts=field_shift, dims=3)
        result[:, :, 0::2, :] = shifted[:, :, 0::2, :]

    return result


# ═══════════════════════════════════════════════════════════════
# Overshoot / Undershoot (Edge Ringing from Re-sharpening)
# ═══════════════════════════════════════════════════════════════


def overshoot_pt(
    tensor: Tensor,
    amount: float = 1.5,
    cutoff: float = 0.4,
    order: int = 2,
) -> Tensor:
    """Simulate edge overshoot/undershoot from aggressive sharpening.

    Models the warp-sharp / edge-enhancement filters commonly applied during
    DVD/broadcast mastering. Boosts frequencies above the Butterworth cutoff.

    Args:
        tensor: BCHW float32 [0,1] tensor.
        amount: Sharpening strength.
        cutoff: Butterworth cutoff as fraction of Nyquist.
        order: Butterworth filter order.

    Returns:
        BCHW float32 [0,1] tensor with overshoot/undershoot artifacts.
    """
    _b, _c, h, w = tensor.shape
    device = tensor.device

    cutoff_freq = max(cutoff * 0.5, 1e-6)
    freq_y = torch.fft.fftfreq(h, device=device, dtype=torch.float32)
    freq_x = torch.fft.rfftfreq(w, device=device, dtype=torch.float32)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing="ij")
    dist = torch.sqrt(fy * fy + fx * fx)
    butterworth = 1.0 / (1.0 + (dist / cutoff_freq) ** (2 * order))
    butterworth = butterworth.unsqueeze(0).unsqueeze(0)

    boost = 1.0 + amount * (1.0 - butterworth)

    spectrum = torch.fft.rfft2(tensor)
    result = torch.fft.irfft2(spectrum * boost, s=(h, w))

    return result.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════
# Temporal Ghosting
# ═══════════════════════════════════════════════════════════════


def temporal_ghosting_pt(
    tensor: Tensor,
    shift_x: int = 3,
    shift_y: int = 0,
    opacity: float = 0.15,
) -> Tensor:
    """Simulate temporal ghosting from residual frame blending.

    Blends a shifted copy of the image at low opacity to simulate
    motion compensation failures and analog signal persistence.

    Args:
        tensor: BCHW float32 [0,1] tensor.
        shift_x: Horizontal ghost displacement in pixels.
        shift_y: Vertical ghost displacement in pixels.
        opacity: Ghost blend strength (0 = invisible, 1 = full double).

    Returns:
        BCHW float32 [0,1] tensor with ghosting artifact.
    """
    if opacity <= 0 or (shift_x == 0 and shift_y == 0):
        return tensor

    ghost = torch.roll(tensor, shifts=(shift_y, shift_x), dims=(2, 3))
    result = tensor * (1.0 - opacity) + ghost * opacity

    return result.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════
# Scanline Darkening (CRT)
# ═══════════════════════════════════════════════════════════════


def scanline_pt(
    tensor: Tensor,
    strength: float = 0.3,
    even_lines: bool = True,
) -> Tensor:
    """Simulate CRT scanline darkening.

    Darkens alternating rows to simulate CRT display scanline gaps.

    Args:
        tensor: BCHW float32 [0,1] tensor.
        strength: Darkening amount (0 = no effect, 1 = fully black scanlines).
        even_lines: If True, darken even rows.

    Returns:
        BCHW float32 [0,1] tensor with scanline artifacts.
    """
    if strength <= 0:
        return tensor

    result = tensor.clone()
    factor = 1.0 - strength

    if even_lines:
        result[:, :, 0::2, :] = result[:, :, 0::2, :] * factor
    else:
        result[:, :, 1::2, :] = result[:, :, 1::2, :] * factor

    return result


# ═══════════════════════════════════════════════════════════════
# Detail Mask Neo (PyTorch port of vsmasktools.detail_mask_neo)
# ═══════════════════════════════════════════════════════════════

_LUMA_R, _LUMA_G, _LUMA_B = 0.2126, 0.7152, 0.0722

_PREWITT_GX = torch.tensor(
    [
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ],
    dtype=torch.float32,
).reshape(1, 1, 3, 3)

_PREWITT_GY = torch.tensor(
    [
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=torch.float32,
).reshape(1, 1, 3, 3)


def _vs_deflate(x: Tensor) -> Tensor:
    """VS std.Deflate for float: pixel = min(pixel, average_of_8_neighbors)."""
    avg = F.avg_pool2d(F.pad(x, [1, 1, 1, 1], mode="reflect"), 3, stride=1)
    avg_8 = (avg * 9.0 - x) / 8.0
    return torch.min(x, avg_8)


def _vs_inflate(x: Tensor) -> Tensor:
    """VS std.Inflate for float: pixel = max(pixel, average_of_8_neighbors)."""
    avg = F.avg_pool2d(F.pad(x, [1, 1, 1, 1], mode="reflect"), 3, stride=1)
    avg_8 = (avg * 9.0 - x) / 8.0
    return torch.max(x, avg_8)


def _bilateral_filter(
    clip: Tensor,
    ref: Tensor,
    sigma_s: float = 1.0,
    sigma_r: float = 0.02,
) -> Tensor:
    """Bilateral filter on grayscale (B, 1, H, W) with reference clip."""
    radius = max(1, int(sigma_s * 3.0 + 0.5))
    ksize = 2 * radius + 1

    coords = torch.arange(ksize, device=clip.device, dtype=torch.float32) - radius
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    spatial_w = torch.exp(-(gx * gx + gy * gy) / (2.0 * sigma_s * sigma_s))

    pad = radius
    ref_pad = F.pad(ref, [pad, pad, pad, pad], mode="reflect")
    clip_pad = F.pad(clip, [pad, pad, pad, pad], mode="reflect")

    b, c, h, w = clip.shape

    ref_unfold = ref_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    clip_unfold = clip_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)

    ref_center = ref.unsqueeze(-1).unsqueeze(-1)
    range_diff = ref_unfold - ref_center
    range_w = torch.exp(-(range_diff * range_diff) / (2.0 * sigma_r * sigma_r))

    spatial_w = spatial_w.reshape(1, 1, 1, 1, ksize, ksize)
    weight = spatial_w * range_w

    numerator = (weight * clip_unfold).sum(dim=(-2, -1))
    denominator = weight.sum(dim=(-2, -1))

    return numerator / (denominator + 1e-10)


def _remove_grain_17(x: Tensor) -> Tensor:
    """RemoveGrain mode 17 (MINMAX_MEDIAN_OPP)."""
    padded = F.pad(x, [1, 1, 1, 1], mode="reflect")
    tl = padded[:, :, :-2, :-2]
    t = padded[:, :, :-2, 1:-1]
    tr = padded[:, :, :-2, 2:]
    l_ = padded[:, :, 1:-1, :-2]
    r = padded[:, :, 1:-1, 2:]
    bl = padded[:, :, 2:, :-2]
    b_ = padded[:, :, 2:, 1:-1]
    br = padded[:, :, 2:, 2:]

    pairs_min = torch.stack(
        [
            torch.min(tl, br),
            torch.min(t, b_),
            torch.min(tr, bl),
            torch.min(l_, r),
        ],
        dim=0,
    )

    pairs_max = torch.stack(
        [
            torch.max(tl, br),
            torch.max(t, b_),
            torch.max(tr, bl),
            torch.max(l_, r),
        ],
        dim=0,
    )

    lo = pairs_min.max(dim=0).values
    hi = pairs_max.min(dim=0).values

    return x.clamp(min=lo, max=hi)


@torch.no_grad()
def detail_mask_neo_pt(
    tensor: Tensor,
    sigma: float = 1.0,
    detail_brz: float = 0.05,
    lines_brz: float = 0.08,
) -> Tensor:
    """PyTorch port of vsmasktools.detail_mask_neo.

    Generates a detail/edge mask from an RGB image. Output is a single-channel
    mask where 1=detail/edge, 0=flat region.

    Args:
        tensor: BCHW float32 [0,1] RGB tensor.
        sigma: Bilateral filter spatial sigma.
        detail_brz: Binarize threshold for detail component.
        lines_brz: Binarize threshold for edge/lines component.

    Returns:
        (B, 1, H, W) float32 mask in [0, 1].
    """
    b, c, h, w = tensor.shape
    device = tensor.device

    luma = (
        _LUMA_R * tensor[:, 0:1]
        + _LUMA_G * tensor[:, 1:2]
        + _LUMA_B * tensor[:, 2:3]
    )

    gs = sigma * 0.75
    grad = max(1, int(gs * 3.0 + 0.5))
    gksize = 2 * grad + 1
    gx = torch.arange(gksize, device=device, dtype=torch.float32) - grad
    gauss_1d = torch.exp(-0.5 * (gx / gs) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    blur_pf = F.conv2d(
        F.pad(luma, [grad, grad, 0, 0], mode="reflect"),
        gauss_1d.reshape(1, 1, 1, gksize),
    )
    blur_pf = F.conv2d(
        F.pad(blur_pf, [0, 0, grad, grad], mode="reflect"),
        gauss_1d.reshape(1, 1, gksize, 1),
    )

    blur_pref = _bilateral_filter(luma, blur_pf, sigma_s=sigma, sigma_r=0.02)

    blur_pref_diff = blur_pref - luma
    blur_pref_diff = _vs_deflate(blur_pref_diff)

    for _ in range(4):
        blur_pref_diff = _vs_inflate(blur_pref_diff)

    gx_kernel = _PREWITT_GX.to(device)
    gy_kernel = _PREWITT_GY.to(device)
    luma_pad = F.pad(luma, [1, 1, 1, 1], mode="reflect")
    edge_x = F.conv2d(luma_pad, gx_kernel)
    edge_y = F.conv2d(luma_pad, gy_kernel)
    prew_mask = torch.sqrt(edge_x * edge_x + edge_y * edge_y)

    prew_mask = _vs_deflate(prew_mask)
    prew_mask = _vs_inflate(prew_mask)

    if detail_brz > 0:
        blur_pref_diff = (blur_pref_diff >= detail_brz).float()
    if lines_brz > 0:
        prew_mask = (prew_mask >= lines_brz).float()

    merged = (blur_pref_diff + prew_mask).clamp(0, 1)

    return _remove_grain_17(merged)


# ═══════════════════════════════════════════════════════════════
# Bicubic Descale (inverse of upscale)
# ═══════════════════════════════════════════════════════════════

_descale_matrix_cache: dict = {}


def _bicubic_weight(t: float) -> float:
    """Keys bicubic kernel with a=-0.75 (matches PyTorch F.interpolate)."""
    t = abs(t)
    a = -0.75
    if t < 1.0:
        return (a + 2.0) * t * t * t - (a + 3.0) * t * t + 1.0
    if t < 2.0:
        return a * t * t * t - 5.0 * a * t * t + 8.0 * a * t - 4.0 * a
    return 0.0


def _build_upscale_matrix(
    src_size: int, dst_size: int, device: torch.device
) -> Tensor:
    """Build the (dst_size, src_size) bicubic upscale weight matrix."""
    A = torch.zeros(dst_size, src_size, device=device, dtype=torch.float64)
    for j in range(dst_size):
        center = (j + 0.5) * src_size / dst_size - 0.5
        i_start = int(center) - 1
        row_sum = 0.0
        for k in range(4):
            i = i_start + k
            w = _bicubic_weight(center - i)
            ii = max(0, min(src_size - 1, i))
            A[j, ii] += w
            row_sum += w
        if row_sum > 0:
            A[j] /= row_sum
    return A


def _build_descale_matrix(
    src_size: int, dst_size: int, device: torch.device
) -> Tensor:
    """Build the (src_size, dst_size) descale matrix = (A^T A)^{-1} A^T."""
    A = _build_upscale_matrix(src_size, dst_size, device)
    AtA = A.T @ A
    AtA_inv = torch.linalg.inv(AtA)
    D = AtA_inv @ A.T
    return D.float()


def _optimal_pad(src_size: int, dst_size: int) -> tuple[int, int]:
    """Find pad_src in [4..40] that minimizes rounding error."""
    best_src, best_dst, best_err = 4, round(4 * dst_size / src_size), float("inf")
    for ps in range(4, 41):
        pd = round(ps * dst_size / src_size)
        err = abs(pd * src_size / dst_size - ps)
        if err < best_err:
            best_err = err
            best_src = ps
            best_dst = pd
    return best_src, best_dst


def _get_descale_matrices(
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, int, int, int, int]:
    """Get cached descale matrices + optimal padding for a resolution pair."""
    cache_key = (src_h, src_w, dst_h, dst_w)
    if (
        cache_key not in _descale_matrix_cache
        or _descale_matrix_cache[cache_key][0].device != device
    ):
        pad_src_h, pad_dst_h = _optimal_pad(src_h, dst_h)
        pad_src_w, pad_dst_w = _optimal_pad(src_w, dst_w)

        D_rows = _build_descale_matrix(
            src_w + 2 * pad_src_w, dst_w + 2 * pad_dst_w, device
        )
        D_cols = _build_descale_matrix(
            src_h + 2 * pad_src_h, dst_h + 2 * pad_dst_h, device
        )

        _descale_matrix_cache[cache_key] = (
            D_rows,
            D_cols,
            pad_src_h,
            pad_dst_h,
            pad_src_w,
            pad_dst_w,
        )

    return _descale_matrix_cache[cache_key]


def bicubic_descale(image: Tensor, orig_h: int, orig_w: int) -> Tensor:
    """Descale a BCHW tensor back to (orig_h, orig_w) by inverting bicubic upscale.

    Only applies descale on axes that were upscaled (orig < current).
    For axes that were downscaled (orig >= current), uses standard bicubic resize.
    """
    b, c, dst_h, dst_w = image.shape

    descale_w = orig_w < dst_w
    descale_h = orig_h < dst_h

    if not descale_w and not descale_h:
        return F.interpolate(
            image, size=(orig_h, orig_w), mode="bicubic", align_corners=False
        )

    result = image

    if descale_w or descale_h:
        D_rows, D_cols, pad_src_h, pad_dst_h, pad_src_w, pad_dst_w = (
            _get_descale_matrices(orig_h, orig_w, dst_h, dst_w, image.device)
        )

    if descale_w:
        cur_h = result.shape[2]
        padded = F.pad(result, [pad_dst_w, pad_dst_w, 0, 0], mode="reflect")
        pw = padded.shape[-1]
        flat = padded.reshape(b * c * cur_h, pw)
        descaled = (flat @ D_rows.T).reshape(b, c, cur_h, orig_w + 2 * pad_src_w)
        result = descaled[:, :, :, pad_src_w : pad_src_w + orig_w]
    else:
        result = F.interpolate(
            result,
            size=(result.shape[2], orig_w),
            mode="bicubic",
            align_corners=False,
        )

    if descale_h:
        cur_w = result.shape[3]
        padded = F.pad(result, [0, 0, pad_dst_h, pad_dst_h], mode="reflect")
        ph = padded.shape[2]
        flat = padded.permute(0, 1, 3, 2).reshape(b * c * cur_w, ph)
        descaled = (flat @ D_cols.T).reshape(b, c, cur_w, orig_h + 2 * pad_src_h)
        result = descaled.permute(0, 1, 3, 2)
        result = result[:, :, pad_src_h : pad_src_h + orig_h, :]
    else:
        result = F.interpolate(
            result,
            size=(orig_h, result.shape[3]),
            mode="bicubic",
            align_corners=False,
        )

    return result


# ═══════════════════════════════════════════════════════════════
# Full NTSC Composite Simulation
# ═══════════════════════════════════════════════════════════════

_NTSC_FSC = 3_579_545.06
_NTSC_SAMPLE_RATE = 14_318_180.24
_NTSC_NYQUIST = _NTSC_SAMPLE_RATE / 2.0
_NTSC_ACTIVE_W = 754
_NTSC_VISIBLE_H = 480
_NTSC_LUMA_BW = 4.2e6
_NTSC_I_BW = 1.5e6
_NTSC_Q_BW = 0.5e6
_NTSC_I_PHASE = 2.147
_NTSC_Q_PHASE = 0.576
_NTSC_COMPOSITE_SCALE = 0.66071429
_NTSC_COMPOSITE_OFFSET = 0.3392857
_NTSC_BLANKING_V = 0.2857
_NTSC_NUM_TAPS = 101

_NTSC_RGB_TO_YIQ = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [0.595901, -0.274557, -0.321344],
        [0.211537, -0.522736, 0.311200],
    ],
    dtype=torch.float32,
)

_NTSC_YIQ_TO_RGB = torch.tensor(
    [
        [1.0, 0.956, 0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106, 1.703],
    ],
    dtype=torch.float32,
)

_ntsc_filter_cache: dict[tuple, Tensor] = {}


def _ntsc_design_fir(
    cutoff_hz: float,
    num_taps: int,
    device: torch.device,
    nyquist: float = _NTSC_NYQUIST,
) -> Tensor:
    """Hamming-windowed sinc lowpass filter."""
    half = (num_taps - 1) / 2.0
    normalized = cutoff_hz / nyquist
    n = torch.arange(num_taps, device=device, dtype=torch.float32) - half
    window = 0.54 - 0.46 * torch.cos(
        2.0
        * 3.141592653589793
        * torch.arange(num_taps, device=device, dtype=torch.float32)
        / (num_taps - 1)
    )
    sinc_vals = torch.sinc(normalized * n)
    kernel = sinc_vals * window
    kernel = kernel / kernel.sum()
    return kernel


def _ntsc_fir_filter_rows(signal: Tensor, kernel: Tensor) -> Tensor:
    """Zero-phase FIR filtering along the last (width) dimension via FFT."""
    b, c, h, w = signal.shape
    num_taps = kernel.shape[0]
    pad_size = num_taps

    padded = F.pad(signal, [pad_size, pad_size, 0, 0], mode="reflect")
    pw = padded.shape[-1]

    fft_n = 1
    while fft_n < pw:
        fft_n *= 2
    sig_flat = padded.reshape(b * c * h, pw)
    S = torch.fft.rfft(sig_flat, n=fft_n)

    cache_key = (round(kernel.sum().item() * 1e8), num_taps, fft_n)
    if (
        cache_key not in _ntsc_filter_cache
        or _ntsc_filter_cache[cache_key].device != signal.device
    ):
        H = torch.fft.rfft(kernel, n=fft_n)
        H_sq = H.real * H.real + H.imag * H.imag
        _ntsc_filter_cache[cache_key] = H_sq
    H_sq = _ntsc_filter_cache[cache_key]

    filtered = torch.fft.irfft(S * H_sq, n=fft_n)
    return filtered[:, pad_size : pad_size + w].reshape(b, c, h, w)


def _ntsc_build_carrier(
    h: int,
    w: int,
    phase_rad: float,
    device: torch.device,
    scale: int = 1,
) -> Tensor:
    """Build NTSC carrier signal (1, 1, H, W)."""
    phase_per_sample = 3.141592653589793 / (2.0 * scale)
    sample_phase = phase_per_sample * torch.arange(
        w, device=device, dtype=torch.float32
    )
    line_phase = 3.141592653589793 * torch.arange(
        h, device=device, dtype=torch.float32
    )
    phase = line_phase.unsqueeze(1) + sample_phase.unsqueeze(0) + phase_rad
    return torch.cos(phase).unsqueeze(0).unsqueeze(0)


def _ntsc_comb_2sample(
    composite: Tensor, scale: int = 1
) -> tuple[Tensor, Tensor]:
    """2-sample delay comb filter for luma/chroma separation."""
    delay = 2 * scale
    delayed = torch.roll(composite, shifts=delay, dims=-1)
    luma = (composite + delayed) * 0.5
    chroma = (composite - delayed) * 0.5
    return luma, chroma


def _ntsc_comb_1h(
    composite: Tensor, scale: int = 1
) -> tuple[Tensor, Tensor]:
    """1H line-delay comb filter."""
    delay = 2 * scale
    ref = torch.roll(composite, shifts=delay, dims=-2)
    luma = (composite + ref) * 0.5
    chroma = (composite - ref) * 0.5
    return luma, chroma


def _ntsc_iir_trailing(signal: Tensor, strength: float) -> Tensor:
    """Causal 1-pole IIR (tape trailing) applied per row.

    Uses CUDA kernel when available, falls back to Python loop.
    """
    if strength <= 0:
        return signal

    if _HAS_IIR_CUDA and signal.is_cuda and iir_trailing_cuda is not None:
        try:
            return iir_trailing_cuda(signal, strength)
        except Exception as exc:
            logger.warning("IIR CUDA kernel failed, using fallback: %s", exc)

    alpha = 1.0 - min(max(strength, 0.0), 1.0) * 0.70
    out = signal.clone()
    for i in range(1, signal.shape[-1]):
        out[..., i] = alpha * out[..., i] + (1.0 - alpha) * out[..., i - 1]
    return out


@torch.no_grad()
def ntsc_composite_pt(
    tensor: Tensor,
    noise: float = 0.05,
    luma_noise: float = 0.0,
    ghost_amplitude: float = 0.0,
    ghost_delay_us: float = 1.5,
    ghost_phase: float = 180.0,
    jitter: float = 0.0,
    edge_ringing: float = 0.0,
    vhs_luma_bw: float = 4.2,
    color_under_bw: float = 500.0,
    tape_trailing: float = 0.0,
    intensity: float = 1.0,
    comb_mode: str = "2sample",
    enable_vhs: bool = False,
) -> Tensor:
    """Full NTSC composite encode/effects/decode at real sample rate.

    Upsamples to N×(754×480) where N is the smallest power-of-2 that makes
    the NTSC resolution >= input resolution. Descales back via bicubic inverse.

    Args:
        tensor: BCHW float32 [0,1] RGB tensor.
        noise: Gaussian noise amplitude (0-0.3).
        luma_noise: Luminance-dependent noise amplitude (0-0.15).
        ghost_amplitude: Multipath ghost strength (0-0.5).
        ghost_delay_us: Ghost delay in microseconds (0.5-10).
        ghost_phase: Ghost phase in degrees (0-360).
        jitter: Per-line timing jitter in subcarrier cycles std dev (0-3).
        edge_ringing: Unsharp mask gain for Gibbs ringing (0-3).
        vhs_luma_bw: VHS luma bandwidth in MHz (1.5-4.2).
        color_under_bw: VHS color-under bandwidth in kHz (200-600).
        tape_trailing: Tape trailing IIR strength (0-1).
        intensity: Blend with original (0=none, 1=full).
        comb_mode: "2sample" or "1h" comb filter.
        enable_vhs: Enable VHS color-under + luma BW limiting.

    Returns:
        BCHW float32 [0,1] tensor with NTSC artifacts.
    """
    if intensity <= 0:
        return tensor

    b, _c, orig_h, orig_w = tensor.shape
    device = tensor.device
    original = tensor

    scale = 1
    while _NTSC_VISIBLE_H * scale < orig_h or _NTSC_ACTIVE_W * scale < orig_w:
        scale *= 2
    ntsc_h = _NTSC_VISIBLE_H * scale
    ntsc_w = _NTSC_ACTIVE_W * scale
    nyquist = _NTSC_NYQUIST * scale
    num_taps = _NTSC_NUM_TAPS * scale

    x = F.interpolate(
        tensor, size=(ntsc_h, ntsc_w), mode="bicubic", align_corners=False
    )

    m_fwd = _NTSC_RGB_TO_YIQ.to(device)
    m_inv = _NTSC_YIQ_TO_RGB.to(device)
    flat = x.reshape(b, 3, ntsc_h * ntsc_w)
    yiq = torch.matmul(m_fwd, flat).reshape(b, 3, ntsc_h, ntsc_w)
    y_ch = yiq[:, 0:1]
    i_ch = yiq[:, 1:2]
    q_ch = yiq[:, 2:3]

    fir_y = _ntsc_design_fir(_NTSC_LUMA_BW, num_taps, device, nyquist)
    fir_i = _ntsc_design_fir(_NTSC_I_BW, num_taps, device, nyquist)
    fir_q = _ntsc_design_fir(_NTSC_Q_BW, num_taps, device, nyquist)

    y_ch = _ntsc_fir_filter_rows(y_ch, fir_y)
    i_ch = _ntsc_fir_filter_rows(i_ch, fir_i)
    q_ch = _ntsc_fir_filter_rows(q_ch, fir_q)

    carrier_i = _ntsc_build_carrier(ntsc_h, ntsc_w, _NTSC_I_PHASE, device, scale)
    carrier_q = _ntsc_build_carrier(ntsc_h, ntsc_w, _NTSC_Q_PHASE, device, scale)

    composite = y_ch + i_ch * carrier_i + q_ch * carrier_q
    composite = composite * _NTSC_COMPOSITE_SCALE + _NTSC_COMPOSITE_OFFSET
    composite = _ntsc_fir_filter_rows(composite, fir_y)

    # === Effects ===

    if enable_vhs:
        vhs_luma, vhs_chroma = _ntsc_comb_2sample(composite, scale)
        lut_period = 4 * scale
        cos_lut = torch.cos(
            (3.141592653589793 / 2.0)
            * torch.arange(lut_period, device=device, dtype=torch.float32)
            / scale
        )
        nsin_lut = -torch.sin(
            (3.141592653589793 / 2.0)
            * torch.arange(lut_period, device=device, dtype=torch.float32)
            / scale
        )
        col_idx = torch.arange(ntsc_w, device=device) % lut_period
        cos_carrier = cos_lut[col_idx].reshape(1, 1, 1, ntsc_w)
        nsin_carrier = nsin_lut[col_idx].reshape(1, 1, 1, ntsc_w)

        i_bb = 2.0 * vhs_chroma * cos_carrier
        q_bb = 2.0 * vhs_chroma * nsin_carrier

        vhs_taps = 61 * scale
        fir_cu = _ntsc_design_fir(color_under_bw * 1000.0, vhs_taps, device, nyquist)
        i_bb = _ntsc_fir_filter_rows(i_bb, fir_cu)
        q_bb = _ntsc_fir_filter_rows(q_bb, fir_cu)

        vhs_chroma = i_bb * cos_carrier + q_bb * nsin_carrier

        fir_vhs_y = _ntsc_design_fir(
            vhs_luma_bw * 1e6, vhs_taps, device, nyquist
        )
        vhs_luma = _ntsc_fir_filter_rows(vhs_luma, fir_vhs_y)

        composite = vhs_luma + vhs_chroma

    if edge_ringing > 0:
        luma_tmp, _ = _ntsc_comb_2sample(composite, scale)
        fir_detail = _ntsc_design_fir(1.5e6, 91 * scale, device, nyquist)
        blurred = _ntsc_fir_filter_rows(luma_tmp, fir_detail)
        composite = composite + edge_ringing * (luma_tmp - blurred)

    if luma_noise > 0:
        luma_level = ((composite - _NTSC_BLANKING_V) / 0.66).clamp(0, 1)
        noise_scale = 1.0 - 0.7 * luma_level
        raw_noise = torch.randn_like(composite)
        fir_noise = _ntsc_design_fir(3.0e6, 31 * scale, device, nyquist)
        filtered_noise = _ntsc_fir_filter_rows(raw_noise, fir_noise)
        composite = composite + luma_noise * noise_scale * filtered_noise

    if noise > 0:
        composite = composite + noise * torch.randn_like(composite)

    if ghost_amplitude > 0:
        sample_rate = _NTSC_SAMPLE_RATE * scale
        delay_samples = ghost_delay_us * sample_rate / 1e6
        delay_int = int(delay_samples)
        frac = delay_samples - delay_int
        ghost = (1.0 - frac) * torch.roll(
            composite, shifts=delay_int, dims=-1
        ) + frac * torch.roll(composite, shifts=delay_int + 1, dims=-1)
        phase_gain = float(
            torch.cos(
                torch.tensor(ghost_phase * 3.141592653589793 / 180.0)
            )
        )
        composite = composite + ghost_amplitude * phase_gain * ghost

    if jitter > 0:
        samples_per_cycle = 4 * scale
        shifts = (torch.randn(ntsc_h, device=device) * jitter).round().long()
        shift_amounts = shifts.abs() * samples_per_cycle
        col_idx = torch.arange(ntsc_w, device=device)
        gather_idx = (
            col_idx.unsqueeze(0) - shift_amounts.unsqueeze(1)
        ) % ntsc_w
        gather_idx = gather_idx.unsqueeze(0).unsqueeze(0).expand_as(composite)
        composite = torch.gather(composite, dim=-1, index=gather_idx)

    if tape_trailing > 0:
        trail_luma, trail_chroma = _ntsc_comb_2sample(composite, scale)
        trail_luma = _ntsc_iir_trailing(trail_luma, tape_trailing)
        composite = trail_luma + trail_chroma

    # === Decode ===

    if comb_mode == "1h":
        dec_luma, dec_chroma = _ntsc_comb_1h(composite, scale)
    else:
        dec_luma, dec_chroma = _ntsc_comb_2sample(composite, scale)

    dec_luma = _ntsc_fir_filter_rows(dec_luma, fir_y)
    dec_luma = (dec_luma - _NTSC_COMPOSITE_OFFSET) / _NTSC_COMPOSITE_SCALE

    i_demod = 2.0 * dec_chroma * carrier_i
    q_demod = 2.0 * dec_chroma * carrier_q

    i_decoded = _ntsc_fir_filter_rows(i_demod, fir_i)
    q_decoded = _ntsc_fir_filter_rows(q_demod, fir_q)

    i_decoded = i_decoded / _NTSC_COMPOSITE_SCALE
    q_decoded = q_decoded / _NTSC_COMPOSITE_SCALE

    yiq_decoded = torch.cat([dec_luma, i_decoded, q_decoded], dim=1)
    flat_decoded = yiq_decoded.reshape(b, 3, ntsc_h * ntsc_w)
    rgb_decoded = (
        torch.matmul(m_inv, flat_decoded)
        .reshape(b, 3, ntsc_h, ntsc_w)
        .clamp(0, 1)
    )

    rgb_decoded = torch.roll(rgb_decoded, shifts=-scale, dims=-1)

    if orig_h != ntsc_h or orig_w != ntsc_w:
        rgb_decoded = bicubic_descale(rgb_decoded, orig_h, orig_w)

    if intensity < 1.0:
        rgb_decoded = original + intensity * (rgb_decoded - original)

    return rgb_decoded.clamp(0, 1)
