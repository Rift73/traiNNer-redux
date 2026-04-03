"""OTF degradation functions.

Ported from wtp_dataset_destroyer. Numpy functions take/return
np.ndarray in HWC float32 [0,1] format. GPU functions operate
on BCHW PyTorch tensors directly.
"""

from __future__ import annotations

import io
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

logger = logging.getLogger("traiNNer")

# ── Constants ──────────────────────────────────────────────────────────────────

VIDEO_SUBSAMPLING = {"444": "yuv444p", "422": "yuv422p", "420": "yuv420p"}

SUBSAMPLING_MAP: dict[str, list[float]] = {
    "4:4:4": [1, 1, 1],
    "4:2:2": [1, 1, 0.5],
    "4:1:1": [1, 1, 0.25],
    "4:2:0": [1, 0.5, 0.5],
    "4:1:0": [1, 0.5, 0.25],
    "4:4:0": [1, 0.5, 1],
    "4:2:1": [1, 0.5, 0.5],
    "4:1:2": [1, 1, 0.25],
    "4:1:3": [1, 0.75, 0.25],
}

YUV_MAP: dict[str, str] = {
    "709": "ITU-R BT.709",
    "2020": "ITU-R BT.2020",
    "601": "ITU-R BT.601",
    "240": "SMPTE-240M",
}


# ── Tensor ↔ Numpy Bridge ─────────────────────────────────────────────────────


# ── GPU Denoising ──────────────────────────────────────────────────────────────


@torch.no_grad()
def nlmeans_denoise_pt(
    x: Tensor,
    h: float = 30.0,
    template_size: int = 7,
    search_size: int = 21,
) -> Tensor:
    """GPU Non-Local Means denoising on a BCHW tensor.

    Uses custom CUDA kernel (v3 D-tile + separable box filter) for
    19.6x speedup over pure PyTorch, 70x faster than OpenCV CUDA.

    Args:
        x: BCHW float32 tensor on GPU.
        h: Filter strength on [0,255] scale. Higher = more smoothing.
        template_size: Patch comparison window size (odd).
        search_size: Search window size (odd).

    Returns:
        Denoised BCHW tensor.
    """
    from traiNNer.data.nlmeans_cuda import nlmeans_denoise_cuda

    return nlmeans_denoise_cuda(x, h, template_size, search_size)


# ── Tensor ↔ Numpy Bridge ─────────────────────────────────────────────────────


def apply_per_image(
    tensor: Tensor, fn: Callable[[np.ndarray], np.ndarray]
) -> Tensor:
    """Apply a per-image numpy function to a BCHW GPU tensor.

    Converts to CPU HWC float32, applies fn per image, converts back.
    """
    device = tensor.device
    dtype = tensor.dtype
    batch_np = tensor.detach().cpu().float().numpy()  # BCHW float32
    results = []
    for i in range(batch_np.shape[0]):
        img = batch_np[i].transpose(1, 2, 0)  # CHW → HWC
        out = fn(img)
        results.append(out.transpose(2, 0, 1))  # HWC → CHW
    stacked = np.stack(results)
    return torch.from_numpy(stacked).to(device=device, dtype=dtype)


# Persistent thread pool for parallel per-image ops (WebP encoding)
_PARALLEL_POOL: ThreadPoolExecutor | None = None


def apply_per_image_parallel(
    tensor: Tensor, fn: Callable[[np.ndarray], np.ndarray], max_workers: int = 8
) -> Tensor:
    """Apply a per-image numpy function in parallel using a thread pool.

    Same interface as apply_per_image but uses persistent ThreadPoolExecutor
    for concurrent CPU-bound ops (e.g., WebP encoding via Pillow).
    """
    global _PARALLEL_POOL  # noqa: PLW0603
    if _PARALLEL_POOL is None:
        _PARALLEL_POOL = ThreadPoolExecutor(max_workers=max_workers)

    device = tensor.device
    dtype = tensor.dtype
    batch_np = tensor.detach().cpu().float().numpy()

    def _process(i: int) -> np.ndarray:
        img = batch_np[i].transpose(1, 2, 0)
        out = fn(img)
        return out.transpose(2, 0, 1)

    futures = [_PARALLEL_POOL.submit(_process, i) for i in range(batch_np.shape[0])]
    results = [f.result() for f in futures]
    stacked = np.stack(results)
    return torch.from_numpy(stacked).to(device=device, dtype=dtype)


# ── Compression ────────────────────────────────────────────────────────────────


def compress_webp(img: np.ndarray, quality: int) -> np.ndarray:
    """Compress image using WebP format via Pillow with method=0.

    Uses Pillow instead of cv2 to access libwebp's `method` parameter.
    method=0 is the fastest encoding setting (1.8x faster than cv2 default).

    Args:
        img: HWC float32 [0,1] RGB image.
        quality: WebP quality (1-100).

    Returns:
        HWC float32 [0,1] RGB image after WebP round-trip.
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_u8)
    buf = io.BytesIO()
    pil_img.save(buf, format="WebP", quality=quality, method=0)
    buf.seek(0)
    decoded = np.array(Image.open(buf))
    return decoded.astype(np.float32) / 255.0


def _video_core(
    img: np.ndarray,
    codec: str,
    output_args: list[str],
    video_sampling: list[str],
    container: str = "mpeg",
) -> np.ndarray:
    """Compress a single image through an ffmpeg video codec pipe.

    Sends the image as a raw frame into ffmpeg encoder, then decodes back.
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    height, width, channels = img_u8.shape
    sampling = VIDEO_SUBSAMPLING[np.random.choice(video_sampling)]

    process1 = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-threads",
            "0",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "30",
            "-i",
            "pipe:",
            "-vcodec",
            codec,
            "-an",
            "-f",
            container,
            "-pix_fmt",
            sampling,
        ]
        + output_args
        + ["pipe:"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process1.stdin is not None
    process1.stdin.write(img_u8.tobytes())
    process1.stdin.flush()
    process1.stdin.close()

    process2 = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-threads",
            "0",
            "-f",
            container,
            "-i",
            "pipe:",
            "-pix_fmt",
            "rgb24",
            "-f",
            "image2pipe",
            "-vcodec",
            "rawvideo",
            "pipe:",
        ],
        stdin=process1.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process2.stdout is not None
    raw_frame = process2.stdout.read()[: (height * width * channels)]
    process2.stdout.close()
    if process2.stderr:
        process2.stderr.close()
    process1.wait()
    process2.wait()

    frame_data = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
        (height, width, channels)
    )
    return frame_data.astype(np.float32) / 255.0


def compress_video(
    img: np.ndarray, codec: str, quality: int, video_sampling: list[str]
) -> np.ndarray:
    """Compress image using a video codec via ffmpeg.

    Args:
        img: HWC float32 [0,1] RGB image.
        codec: One of "h264", "hevc", "mpeg2", "mpeg4", "vp9".
        quality: Codec-specific quality value (CRF or qscale).
        video_sampling: List of video subsampling options, e.g. ["444", "422", "420"].

    Returns:
        HWC float32 [0,1] RGB image after codec round-trip.
    """
    codec_map: dict[str, tuple[str, str, list[str]]] = {
        "h264": ("h264", "mpeg", ["-crf", str(quality)]),
        "hevc": (
            "hevc",
            "mpeg",
            ["-crf", str(quality), "-x265-params", "log-level=0"],
        ),
        "mpeg2": (
            "mpeg2video",
            "mpeg",
            [
                "-qscale:v",
                str(quality),
                "-qmax",
                str(quality),
                "-qmin",
                str(quality),
            ],
        ),
        "mpeg4": (
            "mpeg4",
            "mpeg",
            [
                "-qscale:v",
                str(quality),
                "-qmax",
                str(quality),
                "-qmin",
                str(quality),
            ],
        ),
        "vp9": ("libvpx-vp9", "webm", ["-crf", str(quality), "-b:v", "0"]),
    }
    ffmpeg_codec, container, args = codec_map[codec]
    return _video_core(img, ffmpeg_codec, args, video_sampling, container)


# ── Dithering ──────────────────────────────────────────────────────────────────


def _extract_palette(img: np.ndarray, num_colors: int) -> np.ndarray:
    """Extract a content-adaptive color palette using PIL median cut.

    Args:
        img: HWC float32 [0,1] RGB image.
        num_colors: Number of palette colors (2-256).

    Returns:
        Palette as (1, num_colors, C) float32 array for PaletteQuantization.
    """
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        pil_img = Image.fromarray(img_u8, mode="L")
    else:
        pil_img = Image.fromarray(img_u8, mode="RGB")

    quantized = pil_img.quantize(
        colors=num_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )

    palette_raw = quantized.getpalette()
    if palette_raw is None:
        raise ValueError(f"Failed to extract palette with {num_colors} colors")
    channels = 1 if img.ndim == 2 else 3
    # getpalette() always returns 768 values (256×3) but only the first N
    # entries are meaningful. Count actual unique indices used.
    actual_colors = len(set(quantized.getdata()))
    actual_colors = min(actual_colors, num_colors)
    # Ensure we have enough palette data and at least 2 colors
    max_from_palette = len(palette_raw) // channels
    actual_colors = min(actual_colors, max_from_palette)
    actual_colors = max(actual_colors, 2)
    # If palette is too small for even 2 colors, skip dithering
    if len(palette_raw) < actual_colors * channels:
        return None
    palette = np.array(
        palette_raw[: actual_colors * channels], dtype=np.float32
    ).reshape(1, actual_colors, channels) / 255.0
    return palette


def apply_dithering_palette(
    img: np.ndarray,
    dithering_type: str,
    num_colors: int,
    map_size: int = 4,
    history: int = 12,
    decay_ratio: float = 0.5,
) -> np.ndarray:
    """Apply dithering with content-adaptive palette quantization (indexed color).

    Extracts a palette via median cut, then dithers to those colors.
    Like GIF/PNG8 quantization but with proper error diffusion.

    Args:
        img: HWC float32 [0,1] image.
        dithering_type: Algorithm name.
        num_colors: Number of palette colors (2-256).
        map_size: Map size for ordered dithering.
        history: History length for Riemersma dithering.
        decay_ratio: Decay ratio for Riemersma dithering.

    Returns:
        HWC float32 [0,1] dithered image.
    """
    from chainner_ext import (
        DiffusionAlgorithm,
        PaletteQuantization,
        error_diffusion_dither,
        quantize,
        riemersma_dither,
    )

    error_diffusion_map: dict[str, DiffusionAlgorithm] = {
        "floydsteinberg": DiffusionAlgorithm.FloydSteinberg,
        "jarvisjudiceninke": DiffusionAlgorithm.JarvisJudiceNinke,
        "stucki": DiffusionAlgorithm.Stucki,
        "atkinson": DiffusionAlgorithm.Atkinson,
        "burkes": DiffusionAlgorithm.Burkes,
        "sierra": DiffusionAlgorithm.Sierra,
        "tworowsierra": DiffusionAlgorithm.TwoRowSierra,
        "sierralite": DiffusionAlgorithm.SierraLite,
    }

    palette = _extract_palette(img, num_colors)
    if palette is None:
        return img
    pq = PaletteQuantization(palette)

    if dithering_type in error_diffusion_map:
        result = error_diffusion_dither(img, pq, error_diffusion_map[dithering_type])
    elif dithering_type == "riemersma":
        result = riemersma_dither(img, pq, history, decay_ratio)
    elif dithering_type in ("quantize", "order"):
        # ordered_dither only accepts UniformQuantization — fall back to quantize
        result = quantize(img, pq)
    else:
        logger.warning("Unknown dithering type: %s, skipping", dithering_type)
        return img

    return np.squeeze(result)


def apply_dithering(
    img: np.ndarray,
    dithering_type: str,
    quantize_ch: int,
    map_size: int = 4,
    history: int = 12,
    decay_ratio: float = 0.5,
) -> np.ndarray:
    """Apply dithering/quantization to an image.

    Args:
        img: HWC float32 [0,1] image.
        dithering_type: Algorithm name (e.g. "quantize", "floydsteinberg", "order", "riemersma").
        quantize_ch: Number of quantization levels per channel.
        map_size: Map size for ordered dithering.
        history: History length for Riemersma dithering.
        decay_ratio: Decay ratio for Riemersma dithering.

    Returns:
        HWC float32 [0,1] dithered image.
    """
    from chainner_ext import (
        DiffusionAlgorithm,
        UniformQuantization,
        error_diffusion_dither,
        ordered_dither,
        quantize,
        riemersma_dither,
    )

    error_diffusion_map: dict[str, DiffusionAlgorithm] = {
        "floydsteinberg": DiffusionAlgorithm.FloydSteinberg,
        "jarvisjudiceninke": DiffusionAlgorithm.JarvisJudiceNinke,
        "stucki": DiffusionAlgorithm.Stucki,
        "atkinson": DiffusionAlgorithm.Atkinson,
        "burkes": DiffusionAlgorithm.Burkes,
        "sierra": DiffusionAlgorithm.Sierra,
        "tworowsierra": DiffusionAlgorithm.TwoRowSierra,
        "sierralite": DiffusionAlgorithm.SierraLite,
    }

    uq = UniformQuantization(quantize_ch)

    if dithering_type in error_diffusion_map:
        result = error_diffusion_dither(img, uq, error_diffusion_map[dithering_type])
    elif dithering_type == "order":
        result = ordered_dither(img, uq, map_size)
    elif dithering_type == "riemersma":
        result = riemersma_dither(img, uq, history, decay_ratio)
    elif dithering_type == "quantize":
        result = quantize(img, uq)
    else:
        logger.warning("Unknown dithering type: %s, skipping", dithering_type)
        return img

    return np.squeeze(result)


# ── Channel Shift ──────────────────────────────────────────────────────────────


def _shift_single(
    img: np.ndarray, amount_x: int, amount_y: int, fill_value: float
) -> np.ndarray:
    """Shift a 2D image plane by (amount_x, amount_y) pixels."""
    h, w = img.shape[:2]
    m = np.asarray([[1, 0, amount_x], [0, 1, amount_y]], dtype=np.float32)
    return cv.warpAffine(
        img,
        m,
        (w, h),
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(fill_value,),
    )


def _shift_channel_int(
    img: np.ndarray,
    amount_range: list[list[int]],
    fill_value: float,
) -> np.ndarray:
    """Shift by random integer within [[x_lo, x_hi], [y_lo, y_hi]]."""
    ax = 0 if amount_range[0] == [0, 0] else np.random.randint(*amount_range[0])
    ay = 0 if amount_range[1] == [0, 0] else np.random.randint(*amount_range[1])
    if ax == 0 and ay == 0:
        return img
    return _shift_single(img, ax, ay, fill_value)


def _shift_channel_percent(
    img: np.ndarray,
    amount_range: list[list[int]],
    fill_color: float,
) -> np.ndarray:
    """Shift by random percentage of image dimensions."""
    h, w = img.shape[:2]
    ax = (
        0
        if amount_range[0] == [0, 0]
        else int(h * np.random.uniform(*amount_range[0]) / 100)
    )
    ay = (
        0
        if amount_range[1] == [0, 0]
        else int(w * np.random.uniform(*amount_range[1]) / 100)
    )
    if ax == 0 and ay == 0:
        return img
    return _shift_single(img, ax, ay, fill_color)


def apply_shift(
    img: np.ndarray,
    shift_type: str,
    amounts: list[list[list[int]]],
    percent: bool = False,
) -> np.ndarray:
    """Apply per-channel spatial shift in RGB, YUV, or CMYK color space.

    Args:
        img: HWC float32 [0,1] RGB image.
        shift_type: "rgb", "yuv", or "cmyk".
        amounts: Per-channel shift ranges.
            For RGB/YUV: 3 entries, each [[x_lo,x_hi],[y_lo,y_hi]].
            For CMYK: 4 entries.
        percent: If True, amounts are percentages of image dimensions.

    Returns:
        HWC float32 [0,1] shifted image.
    """
    if img.ndim == 2:
        return img

    shift_fn = _shift_channel_percent if percent else _shift_channel_int

    if shift_type == "rgb":
        for c in range(3):
            img[:, :, c] = shift_fn(img[:, :, c], amounts[c], 1.0)
    elif shift_type == "yuv":
        from pepeline import CvtType, cvt_color

        yuv_img = cvt_color(img, CvtType.RGB2YCvCrBt2020)
        for c in range(3):
            yuv_img[:, :, c] = shift_fn(yuv_img[:, :, c], amounts[c], 1.0)
        img = cvt_color(yuv_img, CvtType.YCvCr2RGBBt2020)
    elif shift_type == "cmyk":
        from pepeline import CvtType, cvt_color

        cmyk_img = cvt_color(img, CvtType.RGB2CMYK)
        for c in range(4):
            cmyk_img[:, :, c] = shift_fn(cmyk_img[:, :, c], amounts[c], 0.0)
        img = cvt_color(cmyk_img, CvtType.CMYK2RGB)

    return img


# ── Chroma Subsampling ─────────────────────────────────────────────────────────


def apply_subsampling(
    img: np.ndarray,
    down_alg: str,
    up_alg: str,
    format_str: str,
    blur_sigma: float | None,
    ycbcr_type: str,
) -> np.ndarray:
    """Apply chroma subsampling degradation.

    Converts to YCbCr, downsamples then upsamples the chroma channels,
    optionally blurs them, and converts back to RGB.

    Args:
        img: HWC float32 [0,1] RGB image.
        down_alg: Downscale algorithm name (e.g. "lanczos", "linear", "nearest").
        up_alg: Upscale algorithm name.
        format_str: Subsampling format (e.g. "4:2:0", "4:2:2").
        blur_sigma: Optional Gaussian blur sigma for chroma channels.
        ycbcr_type: YCbCr standard ("601", "709", "2020", "240").

    Returns:
        HWC float32 [0,1] RGB image after subsampling.
    """
    if img.ndim == 2 or img.shape[2] == 1:
        return img

    import colour
    from chainner_ext import ResizeFilter, resize
    from pepeline.pepeline import fast_color_level

    interpolation_map: dict[str, ResizeFilter] = {
        "nearest": ResizeFilter.Nearest,
        "box": ResizeFilter.Box,
        "hermite": ResizeFilter.Hermite,
        "linear": ResizeFilter.Linear,
        "lagrange": ResizeFilter.Lagrange,
        "cubic_catrom": ResizeFilter.CubicCatrom,
        "cubic_mitchell": ResizeFilter.CubicMitchell,
        "cubic_bspline": ResizeFilter.CubicBSpline,
        "lanczos": ResizeFilter.Lanczos,
        "gauss": ResizeFilter.Gauss,
    }

    yuv_key = YUV_MAP[ycbcr_type]
    weights = colour.models.rgb.ycbcr.WEIGHTS_YCBCR[yuv_key]

    lq = colour.RGB_to_YCbCr(img, in_bits=8, K=weights).astype(np.float32)

    scale_list = SUBSAMPLING_MAP[format_str]
    if scale_list != [1, 1, 1]:
        shape = lq.shape
        d = interpolation_map[down_alg]
        u = interpolation_map[up_alg]
        chroma = lq[..., 1:3]
        down_h = int(shape[0] * scale_list[1])
        down_w = int(shape[1] * scale_list[2])
        downsampled = resize(chroma, (down_w, down_h), d, False).squeeze()
        upsampled = resize(downsampled, (shape[1], shape[0]), u, False).squeeze()
        lq[..., 1:3] = fast_color_level(upsampled, 1, 254)

    if blur_sigma is not None and blur_sigma > 0:
        lq[..., 1] = cv.GaussianBlur(
            lq[..., 1],
            (0, 0),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma,
            borderType=cv.BORDER_REFLECT,
        )
        lq[..., 2] = cv.GaussianBlur(
            lq[..., 2],
            (0, 0),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma,
            borderType=cv.BORDER_REFLECT,
        )

    result = colour.YCbCr_to_RGB(lq, in_bits=8, out_bits=8, K=weights)
    return result.astype(np.float32).clip(0, 1)
