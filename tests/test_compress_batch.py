"""Batch video compression (compress_video_batch): output contract, intra-only
coding and the quality setting, for every codec at 8 x 64².
"""

from __future__ import annotations

import contextlib
import io
import os
from typing import Any

import av
import pytest
import torch
import torchcodec.decoders
from av.video.frame import PictureType
from traiNNer.data.compress_video_batch import (
    _CODEC_SUPPORTED_SAMPLING,
    _thread_counts,
    _ThreadCounts,
    compress_video_batch,
)
from traiNNer.utils.rng import RNG

CODECS = ("h264", "hevc", "mpeg4", "mpeg2", "vp9")
QUALITY = {"h264": 30, "hevc": 30, "mpeg4": 10, "mpeg2": 10, "vp9": 35}
# (better, worse) quality settings: CRF for h264/hevc/vp9, qscale for mpeg2/4
QUALITY_PAIR = {
    "h264": (20, 40),
    "hevc": (20, 40),
    "mpeg4": (2, 20),
    "mpeg2": (2, 20),
    "vp9": (20, 50),
}
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]
CODEC_SAMPLING = [
    (codec, sampling)
    for codec in CODECS
    for sampling in sorted(_CODEC_SUPPORTED_SAMPLING[codec])
]


@pytest.fixture(autouse=True, scope="module")
def _project_rng() -> None:
    # The chroma format is drawn from the project RNG.
    with contextlib.suppress(RuntimeError):  # already initialised
        RNG.init_rng(0)


def _frames(batch: int, height: int, width: int, seed: int) -> torch.Tensor:
    """Distinct smooth frames with some fine detail, BCHW float32 in [0, 1]."""
    gen = torch.Generator().manual_seed(seed)
    coarse = torch.rand(batch, 3, 8, 8, generator=gen)
    smooth = torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bicubic", align_corners=False
    )
    detail = torch.rand(batch, 3, height, width, generator=gen)
    return (0.8 * smooth + 0.2 * detail).clamp(0, 1)


def _record_bitstreams(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Record every encoded stream compress_video_batch hands to a decoder."""
    streams: list[bytes] = []
    tc_decoder = torchcodec.decoders.VideoDecoder
    av_open = av.open

    def recording_decoder(source: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        streams.append(source.numpy().tobytes())
        return tc_decoder(source, *args, **kwargs)

    def recording_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if not args and kwargs.get("mode", "r") == "r":
            streams.append(file.getvalue())
        return av_open(file, *args, **kwargs)

    monkeypatch.setattr(torchcodec.decoders, "VideoDecoder", recording_decoder)
    monkeypatch.setattr(av, "open", recording_open)
    return streams


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("codec", "sampling"), CODEC_SAMPLING)
def test_output_contract(codec: str, sampling: str, device: str) -> None:
    x = _frames(8, 64, 64, seed=1).to(device)
    out = compress_video_batch(x, codec, QUALITY[codec], [sampling])

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert out.device == x.device
    assert out.is_contiguous()
    assert out.min() >= 0
    assert out.max() <= 1
    assert not torch.equal(out, x)  # lossy round trip
    # Frame order is kept: each output frame is closest to its own input.
    dist = (out[:, None] - x[None]).square().mean(dim=(2, 3, 4))
    assert torch.equal(dist.argmin(dim=1).cpu(), torch.arange(8))


@pytest.mark.parametrize("codec", CODECS)
def test_odd_size_is_padded_and_cropped(codec: str) -> None:
    x = _frames(8, 63, 65, seed=2)
    out = compress_video_batch(x, codec, QUALITY[codec], ["420"])

    assert out.shape == x.shape
    assert out.is_contiguous()
    assert (out - x).abs().mean() < 0.1


@pytest.mark.parametrize("codec", CODECS)
def test_every_frame_is_intra(codec: str, monkeypatch: pytest.MonkeyPatch) -> None:
    streams = _record_bitstreams(monkeypatch)
    compress_video_batch(_frames(8, 64, 64, seed=3), codec, QUALITY[codec], ["420"])
    monkeypatch.undo()

    assert len(streams) == 1
    with av.open(io.BytesIO(streams[0])) as container:
        frames = [(f.pict_type, f.key_frame) for f in container.decode(video=0)]
    assert frames == [(PictureType.I, True)] * 8


@pytest.mark.parametrize(
    ("codec", "sampling"),
    [(c, s) for c, s in CODEC_SAMPLING if c in ("mpeg4", "mpeg2", "vp9")],
)
def test_identical_frames_decode_identically(codec: str, sampling: str) -> None:
    """Copies of a frame at positions 0 and 4 and at 2 and 7 decode to the same
    output; with inter coding the later copy would be predicted from a
    different neighbour.

    Only codecs with a fixed quantiser: x264 and x265 in CRF mode set each
    frame's QP from the complexity of the frames before it, so identical frames
    at different positions can get different QPs. Their intra-only coding is
    covered by test_every_frame_is_intra.
    """
    x = _frames(8, 64, 64, seed=4)
    x[4] = x[0]
    x[7] = x[2]
    out = compress_video_batch(x, codec, QUALITY[codec], [sampling])

    assert torch.equal(out[0], out[4])
    assert torch.equal(out[2], out[7])
    assert not torch.equal(out[0], out[2])


def test_thread_counts_follow_cpus_and_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(32)), raising=False
    )
    assert _thread_counts(8, 256, 256) == _ThreadCounts(8, 8, 1, 8)
    assert _thread_counts(8, 512, 512) == _ThreadCounts(8, 8, 4, 8)
    assert _thread_counts(3, 64, 64) == _ThreadCounts(8, 3, 1, 8)

    # Pinned to 2 CPUs: the affinity mask wins over the machine's core count.
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert _thread_counts(8, 512, 512) == _ThreadCounts(2, 2, 2, 2)

    # No affinity API (Windows, macOS): os.cpu_count(), or 1 when unknown.
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert _thread_counts(8, 512, 512) == _ThreadCounts(3, 3, 3, 3)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert _thread_counts(8, 512, 512) == _ThreadCounts(1, 1, 1, 1)


@pytest.mark.parametrize("codec", CODECS)
def test_quality_setting_applies(codec: str) -> None:
    x = _frames(8, 64, 64, seed=5)
    better, worse = QUALITY_PAIR[codec]
    err_better = (compress_video_batch(x, codec, better, ["420"]) - x).square().mean()
    err_worse = (compress_video_batch(x, codec, worse, ["420"]) - x).square().mean()

    assert err_worse > 1.5 * err_better
