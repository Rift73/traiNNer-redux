"""Batch video compression — optimized implementations per codec.

Operates on BCHW GPU tensors directly. No apply_per_image bridge.

Strategy (from exhaustive benchmarking):
- TorchCodec BATCH for: libx264, libx265, mpeg4
  (native tensor I/O, single encode call for entire batch)
- PyAV BATCH for: libvpx-vp9 (needs cpu-used option), mpeg2video
  (single container, all frames, in-process)

IMPORTANT: All frames are encoded as intra-only (gop_size=1, no B-frames).
This ensures each frame is independently compressed — batch-as-video is purely
a performance optimization, not a semantic change from per-image encoding.
One exception: x264 and x265 in CRF mode set each frame's QP from the running
complexity of the frames before it in the batch, so the same frame can get a
different QP depending on its neighbours (prediction stays intra).

Thread counts come from _thread_counts. None of them changes the decoded
frames.
"""

from __future__ import annotations

import io
import os
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from traiNNer.utils.rng import RNG

# Codec → TorchCodec format mapping
_TC_CODEC_MAP = {
    "h264": ("libx264", "mp4"),
    "hevc": ("libx265", "mp4"),
    "mpeg4": ("mpeg4", "mp4"),
}

# Codec → PyAV format/options mapping
_PYAV_CODEC_MAP = {
    "vp9": ("libvpx-vp9", "webm"),
    "mpeg2": ("mpeg2video", "mpegts"),
}

# Video subsampling map (matches traiNNer-redux VIDEO_SUBSAMPLING)
_VIDEO_SUBSAMPLING = {
    "444": "yuv444p",
    "422": "yuv422p",
    "420": "yuv420p",
}

# Supported pixel formats per codec (subset of _VIDEO_SUBSAMPLING keys)
# Codecs reject unsupported formats at encode time — filter before selection.
_CODEC_SUPPORTED_SAMPLING: dict[str, set[str]] = {
    "h264": {"444", "422", "420"},
    "hevc": {"444", "422", "420"},
    "mpeg4": {"420"},              # only yuv420p
    "vp9": {"444", "422", "420"},
    "mpeg2": {"422", "420"},       # no yuv444p
}


def _filter_sampling(
    video_sampling: list[str], codec: str
) -> list[str]:
    """Filter sampling options to only those supported by the codec."""
    supported = _CODEC_SUPPORTED_SAMPLING.get(codec)
    if supported is None:
        return video_sampling
    filtered = [s for s in video_sampling if s in supported]
    return filtered if filtered else ["420"]  # fallback to 420 if nothing matches


class _ThreadCounts(NamedTuple):
    x265_pools: int
    x265_frame_threads: int
    vp9_encoder: int
    vp9_decoder: int


def _thread_counts(batch: int, height: int, width: int) -> _ThreadCounts:
    """Encoder and decoder thread counts for one call.

    The codec defaults suit long videos, not a batch of small intra-only
    frames: x265 runs one frame thread next to a pool of one worker per core,
    and libvpx starts one thread per core, whose start-up costs more than it
    saves on frames up to 256². Intra-only frames are independent, so frame
    threads (x265 encode, vp9 decode) spread the batch over the cores.

    The counts derive from the batch size and from the CPUs this process may
    run on: its affinity mask where the OS has one (training may be pinned),
    otherwise os.cpu_count(). On a 16-core/32-thread CPU, batch 8: x265 pool 8
    and 8 frame threads, libvpx 1 thread up to 256² and 4 above, 8 vp9 decode
    threads.
    """
    if hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1
    pools = min(8, cpus)
    return _ThreadCounts(
        x265_pools=pools,
        x265_frame_threads=min(batch, pools),
        vp9_encoder=1 if height * width <= 256 * 256 else min(4, cpus),
        vp9_decoder=min(8, cpus),
    )


def _to_unit_float(frames: Tensor, device: torch.device) -> Tensor:
    """BCHW uint8 CPU frames → contiguous float32 [0, 1] on `device`.

    The frames cross to the device as uint8 (a quarter of the float32 bytes).
    The divisor is a 0-dim device tensor because CUDA turns a Python-scalar
    divisor into a multiply by its reciprocal, which is 1 ulp off true division
    for about half of the values 0..255.
    """
    frames = frames.to(device)
    divisor = torch.full((), 255.0, device=device)
    return frames.to(torch.float32, memory_format=torch.contiguous_format).div_(divisor)


def compress_video_batch_torchcodec(
    tensor: Tensor,
    codec: str = "h264",
    quality: int = 28,
    video_sampling: list[str] | None = None,
) -> Tensor:
    """Batch video compression via TorchCodec BATCH.

    Encodes all batch frames as intra-only video (gop_size=1), decodes back.
    Native tensor I/O — no numpy, no apply_per_image.

    Supported codecs: h264, hevc, mpeg4.

    Args:
        tensor: BCHW float32 tensor on GPU, values in [0, 1].
        codec: "h264", "hevc", or "mpeg4".
        quality: CRF value (lower = better quality).
        video_sampling: List of chroma subsampling options, e.g. ["444", "422", "420"].
            One is randomly chosen per batch. If None, uses codec default (usually yuv420p).

    Returns:
        BCHW float32 tensor on same device, after codec roundtrip.
    """
    from torchcodec.decoders import VideoDecoder
    from torchcodec.encoders import VideoEncoder

    if codec not in _TC_CODEC_MAP:
        raise ValueError(
            f"TorchCodec batch: unsupported codec '{codec}'. "
            f"Use one of: {list(_TC_CODEC_MAP.keys())}"
        )

    device = tensor.device
    codec_name, fmt = _TC_CODEC_MAP[codec]
    b, _, orig_h, orig_w = tensor.shape

    # Chroma subsampling selection (filter to codec-supported formats)
    if video_sampling:
        filtered = _filter_sampling(video_sampling, codec)
        sampling_key = RNG.get_rng().choice(filtered)
        pix_fmt = _VIDEO_SUBSAMPLING.get(sampling_key, "yuv420p")
    else:
        pix_fmt = None  # codec default

    # Pad to even dims if needed by chroma subsampling (trainer generates
    # arbitrary resize dims — odd sizes are expected, not edge cases)
    pad_h = pad_w = 0
    if pix_fmt and pix_fmt != "yuv444p":
        need_w = 2 if "420" in pix_fmt or "422" in pix_fmt else 1
        need_h = 2 if "420" in pix_fmt else 1
        pad_w = (need_w - orig_w % need_w) % need_w
        pad_h = (need_h - orig_h % need_h) % need_h

    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    # BCHW float [0,1] → BCHW uint8 on CPU
    inp = (tensor.clamp(0, 1) * 255).byte().cpu()

    # Encode entire batch as one intra-only video
    enc = VideoEncoder(frames=inp, frame_rate=1)

    # Build encode kwargs — gop=1 ensures every frame is an I-frame
    extra_options: dict[str, str] = {"g": "1", "bf": "0"}
    if codec_name == "libx264":
        extra_options["preset"] = "ultrafast"
    elif codec_name == "libx265":
        threads = _thread_counts(b, orig_h, orig_w)
        extra_options["preset"] = "ultrafast"
        extra_options["x265-params"] = (
            f"log-level=0:pools={threads.x265_pools}"
            f":frame-threads={threads.x265_frame_threads}"
        )

    encode_kwargs: dict = {
        "format": fmt,
        "codec": codec_name,
        "extra_options": extra_options,
    }
    if codec_name == "mpeg4":
        # The mpeg4 encoder has no CRF option and ignores `crf`; its quality is
        # a fixed qscale, as in compress_video.
        extra_options["qmin"] = extra_options["qmax"] = str(quality)
    else:
        encode_kwargs["crf"] = quality

    if pix_fmt:
        encode_kwargs["pixel_format"] = pix_fmt

    encoded = enc.to_tensor(**encode_kwargs)

    # Decode all frames back
    dec = VideoDecoder(encoded)
    frames = torch.stack([dec[i] for i in range(len(dec))])

    # Crop padding, convert back to float on original device
    if pad_h or pad_w:
        frames = frames[:, :, :orig_h, :orig_w]
    return _to_unit_float(frames, device)


def compress_video_batch_pyav(
    tensor: Tensor,
    codec: str = "vp9",
    quality: int = 28,
    video_sampling: list[str] | None = None,
) -> Tensor:
    """Batch video compression via PyAV (single container, all frames).

    Uses PyAV for codecs that need special options (VP9 speed) or that
    TorchCodec can't decode (MPEG2). All frames are intra-only (gop_size=1).

    Supported codecs: vp9, mpeg2.

    Args:
        tensor: BCHW float32 tensor on GPU, values in [0, 1].
        codec: "vp9" or "mpeg2".
        quality: CRF/quality value.
        video_sampling: List of chroma subsampling options, e.g. ["444", "422", "420"].
            One is randomly chosen per batch. If None, defaults to "yuv420p".

    Returns:
        BCHW float32 tensor on same device, after codec roundtrip.
    """
    import av
    from av.video.reformatter import VideoReformatter

    if codec not in _PYAV_CODEC_MAP:
        raise ValueError(
            f"PyAV batch: unsupported codec '{codec}'. "
            f"Use one of: {list(_PYAV_CODEC_MAP.keys())}"
        )

    device = tensor.device
    codec_name, fmt = _PYAV_CODEC_MAP[codec]
    b, _, h, w = tensor.shape

    # BCHW float → numpy uint8
    inp_np = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()

    # Chroma subsampling (filter to codec-supported formats)
    if video_sampling:
        filtered = _filter_sampling(video_sampling, codec)
        sampling_key = RNG.get_rng().choice(filtered)
        pix_fmt = _VIDEO_SUBSAMPLING.get(sampling_key, "yuv420p")
    else:
        pix_fmt = "yuv420p"

    # Pad to even dims if needed by chroma subsampling
    pad_h = h % 2 if "420" in pix_fmt else 0
    pad_w = w % 2 if "420" in pix_fmt or "422" in pix_fmt else 0
    if pad_h or pad_w:
        inp_np = np.pad(
            inp_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
        )
        _, _, h_padded, w_padded = inp_np.shape
    else:
        h_padded, w_padded = h, w

    # Build codec options and encoder thread count
    threads = _thread_counts(b, h_padded, w_padded)
    if codec == "vp9":
        options = {"cpu-used": "8", "crf": str(quality), "row-mt": "1"}
        enc_threads = threads.vp9_encoder
    else:
        # Fixed qscale. The slice encoder's default of one thread per core
        # (capped at 16) is several times slower than 1 thread on these frames.
        options = {"qmax": str(quality), "qmin": str(quality)}
        enc_threads = 1

    # Encode all frames into single intra-only container
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=fmt)
    stream = container.add_stream(codec_name, rate=1)
    stream.width = w_padded
    stream.height = h_padded
    stream.pix_fmt = pix_fmt
    stream.gop_size = 1  # Every frame is an I-frame
    stream.options = options
    stream.codec_context.thread_count = enc_threads

    for i in range(b):
        frame_hwc = inp_np[i].transpose(1, 2, 0)  # CHW → HWC
        vf = av.VideoFrame.from_ndarray(frame_hwc, format="rgb24")
        for pkt in stream.encode(vf):
            container.mux(pkt)
    for pkt in stream.encode(None):
        container.mux(pkt)
    container.close()

    # Decode all frames back
    buf.seek(0)
    dec_container = av.open(buf)
    dec_ctx = dec_container.streams.video[0].codec_context
    if codec == "vp9":
        # Intra-only frames decode in parallel on frame threads.
        dec_ctx.thread_type = "FRAME"
        dec_ctx.thread_count = threads.vp9_decoder
    else:
        dec_ctx.thread_count = 1
    # One converter for the batch: frame.to_ndarray(format=...) builds a new
    # swscale context for every frame.
    to_rgb = VideoReformatter()
    frames = [
        to_rgb.reformat(frame, format="rgb24").to_ndarray()[:h, :w]  # Crop padding
        for frame in dec_container.decode(video=0)
    ]
    dec_container.close()

    # HWC → CHW, stack, convert to float on original device
    return _to_unit_float(
        torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2), device
    )


def compress_video_batch(
    tensor: Tensor,
    codec: str = "h264",
    quality: int = 28,
    video_sampling: list[str] | None = None,
) -> Tensor:
    """Unified batch video compression — dispatches to best backend per codec.

    All frames are encoded as intra-only (gop_size=1, no B-frames) to ensure
    semantic equivalence with per-image compression. Chroma subsampling is
    randomly sampled from video_sampling per batch.

    Args:
        tensor: BCHW float32 tensor on GPU, values in [0, 1].
        codec: "h264", "hevc", "mpeg4", "vp9", or "mpeg2".
        quality: CRF/quality value.
        video_sampling: List of chroma subsampling options, e.g. ["444", "422", "420"].
            One is randomly chosen per batch. If None, uses codec default.

    Returns:
        BCHW float32 tensor on same device, after codec roundtrip.
    """
    if codec in _TC_CODEC_MAP:
        return compress_video_batch_torchcodec(tensor, codec, quality, video_sampling)
    elif codec in _PYAV_CODEC_MAP:
        return compress_video_batch_pyav(tensor, codec, quality, video_sampling)
    else:
        raise ValueError(
            f"Unknown codec '{codec}'. "
            f"Supported: {list(_TC_CODEC_MAP.keys()) + list(_PYAV_CODEC_MAP.keys())}"
        )
