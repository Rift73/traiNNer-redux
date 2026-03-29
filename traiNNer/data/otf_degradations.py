"""CPU-based OTF degradation functions.

Ported from wtp_dataset_destroyer. All degradation functions take/return
np.ndarray in HWC float32 [0,1] format.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Callable

import cv2 as cv
import numpy as np
import torch
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


# ── Compression ────────────────────────────────────────────────────────────────


def compress_webp(img: np.ndarray, quality: int) -> np.ndarray:
    """Compress image using WebP format.

    Args:
        img: HWC float32 [0,1] RGB image.
        quality: WebP quality (1-100).

    Returns:
        HWC float32 [0,1] RGB image after WebP round-trip.
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv.cvtColor(img_u8, cv.COLOR_RGB2BGR)
    _, encimg = cv.imencode(".webp", img_bgr, [int(cv.IMWRITE_WEBP_QUALITY), quality])
    decoded = cv.imdecode(encimg, 1)
    return cv.cvtColor(decoded, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0


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
