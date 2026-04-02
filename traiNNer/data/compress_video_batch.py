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
"""

from __future__ import annotations

import io

import numpy as np
import torch
from torch import Tensor

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
    "mpeg4": {"444", "422", "420"},
    "vp9": {"444", "422", "420"},
    "mpeg2": {"422", "420"},  # no yuv444p support
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
            "TorchCodec batch: unsupported codec '{}'. Use one of: {}".format(
                codec, list(_TC_CODEC_MAP.keys())
            )
        )

    device = tensor.device
    codec_name, fmt = _TC_CODEC_MAP[codec]
    _, _, orig_h, orig_w = tensor.shape

    # Chroma subsampling selection (filter to codec-supported formats)
    if video_sampling:
        filtered = _filter_sampling(video_sampling, codec)
        sampling_key = np.random.choice(filtered)
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
        extra_options["preset"] = "ultrafast"
        extra_options["x265-params"] = "log-level=0"

    encode_kwargs: dict = {
        "format": fmt,
        "codec": codec_name,
        "crf": quality,
        "extra_options": extra_options,
    }

    if pix_fmt:
        encode_kwargs["pixel_format"] = pix_fmt

    encoded = enc.to_tensor(**encode_kwargs)

    # Decode all frames back
    dec = VideoDecoder(encoded)
    frames = [dec[i] for i in range(len(dec))]

    # Stack, crop padding, convert back to float on original device
    result = torch.stack(frames).float().div_(255.0).to(device)
    if pad_h or pad_w:
        result = result[:, :, :orig_h, :orig_w]
    return result


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

    if codec not in _PYAV_CODEC_MAP:
        raise ValueError(
            "PyAV batch: unsupported codec '{}'. Use one of: {}".format(
                codec, list(_PYAV_CODEC_MAP.keys())
            )
        )

    device = tensor.device
    codec_name, fmt = _PYAV_CODEC_MAP[codec]
    b, _, h, w = tensor.shape

    # BCHW float → numpy uint8
    inp_np = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()

    # Chroma subsampling (filter to codec-supported formats)
    if video_sampling:
        filtered = _filter_sampling(video_sampling, codec)
        sampling_key = np.random.choice(filtered)
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

    # Build codec options
    if codec == "vp9":
        options = {"cpu-used": "8", "crf": str(quality), "row-mt": "1"}
    elif codec == "mpeg2":
        options = {
            "qscale:v": str(quality),
            "qmax": str(quality),
            "qmin": str(quality),
        }
    else:
        options = {}

    # Encode all frames into single intra-only container
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=fmt)
    stream = container.add_stream(codec_name, rate=1)
    stream.width = w_padded
    stream.height = h_padded
    stream.pix_fmt = pix_fmt
    stream.gop_size = 1  # Every frame is an I-frame
    stream.options = options

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
    frames = []
    for frame in dec_container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(arr[:h, :w, :]))  # Crop padding
    dec_container.close()

    # HWC → CHW, stack, convert to float
    result = torch.stack([f.permute(2, 0, 1) for f in frames])
    return result.float().div_(255.0).to(device)


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
            "Unknown codec '{}'. Supported: {}".format(
                codec, list(_TC_CODEC_MAP.keys()) + list(_PYAV_CODEC_MAP.keys())
            )
        )
