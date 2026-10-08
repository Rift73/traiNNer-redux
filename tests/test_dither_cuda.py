"""Bit-exactness of the CUDA dithering kernels against chainner_ext."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from traiNNer.data.dither_cuda import (
    DIFFUSION_ALGORITHMS,
    dither_cuda_available,
    error_diffusion_dither_cuda,
    riemersma_dither_cuda,
)

chainner_ext = pytest.importorskip("chainner_ext")

ALGORITHM_ENUM = {
    name.lower(): getattr(chainner_ext.DiffusionAlgorithm, name)
    for name in dir(chainner_ext.DiffusionAlgorithm)
    if not name.startswith("_")
}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SIZES = [(64, 64), (128, 128), (257, 131), (131, 257), (256, 64), (64, 256)]
LEVELS = (2, 3, 7, 16, 64)
BATCH_LEVELS = (2, 3, 7, 16, 64, 7, 3, 2)
HISTORIES = (2, 8, 16)
DECAYS = (0.1, 0.5, 0.9)


def _noise(height: int, width: int, channels: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(
        (height, width, channels), dtype=np.float32
    )


def _gradient(height: int, width: int, channels: int, tilt: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = (1 - tilt) * xx / max(width - 1, 1) + tilt * yy / max(height - 1, 1)
    planes = [base * (1 - 0.25 * c) + 0.1 * c for c in range(channels)]
    return np.stack(planes, axis=-1).astype(np.float32)


def _batch(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Eight HWC images: noise, gradients, and noise outside [0, 1]."""
    images = []
    for i in range(7):
        if i % 2:
            images.append(_gradient(height, width, channels, tilt=0.15 * i))
        else:
            images.append(
                _noise(height, width, channels, seed=height * 1000 + width + i)
            )
    images.append(_noise(height, width, channels, seed=7) * 1.5 - 0.25)
    return np.stack(images)


def _run(
    images: np.ndarray, levels: tuple[int, ...], dither: str, **kwargs: float
) -> np.ndarray:
    """Dither a stack of HWC images on the GPU; returns HWC images."""
    x = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).cuda()
    before = x.clone()
    per_sample = torch.tensor(levels, dtype=torch.int64)
    if dither == "riemersma":
        out = riemersma_dither_cuda(
            x, per_sample, int(kwargs["history"]), float(kwargs["decay_ratio"])
        )
    else:
        out = error_diffusion_dither_cuda(x, per_sample, dither)
    assert torch.equal(x, before), "the input tensor was modified"
    return out.cpu().numpy().transpose(0, 2, 3, 1)


def _reference(
    image: np.ndarray, levels: int, dither: str, **kwargs: float
) -> np.ndarray:
    quant = chainner_ext.UniformQuantization(levels)
    if dither == "riemersma":
        result = chainner_ext.riemersma_dither(
            image, quant, int(kwargs["history"]), float(kwargs["decay_ratio"])
        )
    else:
        result = chainner_ext.error_diffusion_dither(
            image, quant, ALGORITHM_ENUM[dither]
        )
    return np.asarray(result, dtype=np.float32).reshape(image.shape)


def _check(
    images: np.ndarray, levels: tuple[int, ...], dither: str, **kwargs: float
) -> None:
    actual = _run(images, levels, dither, **kwargs)
    for i, (image, level) in enumerate(zip(images, levels, strict=True)):
        expected = _reference(np.ascontiguousarray(image), level, dither, **kwargs)
        got = np.ascontiguousarray(actual[i])
        mismatched = got.view(np.uint32) != expected.view(np.uint32)
        if mismatched.any():
            diff = float(np.abs(got.astype(np.float64) - expected).max())
            pytest.fail(
                f"{dither} {kwargs} image {i} {image.shape} levels {level}: "
                f"{int(mismatched.sum())} values differ, max abs diff {diff}"
            )


def test_extension_builds() -> None:
    assert dither_cuda_available()


def test_algorithm_ids_follow_chainner_ext() -> None:
    ids = {name: int(ALGORITHM_ENUM[name]) for name in DIFFUSION_ALGORITHMS}
    assert ids == DIFFUSION_ALGORITHMS


@pytest.mark.parametrize("algorithm", list(DIFFUSION_ALGORITHMS))
@pytest.mark.parametrize(("height", "width"), SIZES)
def test_error_diffusion_bit_exact(height: int, width: int, algorithm: str) -> None:
    images = _batch(height, width)
    _check(images, BATCH_LEVELS, algorithm)
    for level in LEVELS:
        for image in images[:2]:
            _check(image[None], (level,), algorithm)


@pytest.mark.parametrize("decay_ratio", DECAYS)
@pytest.mark.parametrize("history", HISTORIES)
@pytest.mark.parametrize(("height", "width"), SIZES)
def test_riemersma_bit_exact(
    height: int, width: int, history: int, decay_ratio: float
) -> None:
    images = _batch(height, width)
    _check(images, BATCH_LEVELS, "riemersma", history=history, decay_ratio=decay_ratio)
    for level in LEVELS:
        for image in images[:2]:
            _check(
                image[None],
                (level,),
                "riemersma",
                history=history,
                decay_ratio=decay_ratio,
            )


@pytest.mark.parametrize("channels", [1, 4])
@pytest.mark.parametrize("dither", [*DIFFUSION_ALGORITHMS, "riemersma"])
def test_channel_counts_bit_exact(dither: str, channels: int) -> None:
    images = _batch(61, 67, channels)
    _check(images, BATCH_LEVELS, dither, history=12, decay_ratio=0.3)


@pytest.mark.parametrize(
    ("height", "width", "history"), [(131, 257, 33), (17, 19, 13000)]
)
def test_riemersma_long_history_bit_exact(
    height: int, width: int, history: int
) -> None:
    """Histories past the register kernels: shared-memory and global-scratch rings."""
    _check(
        _batch(height, width),
        BATCH_LEVELS,
        "riemersma",
        history=history,
        decay_ratio=0.5,
    )


@pytest.mark.parametrize("algorithm", ["floydsteinberg", "stucki"])
def test_error_diffusion_wide_image_bit_exact(algorithm: str) -> None:
    """Rows wider than the shared-memory ring use the global-scratch ring."""
    _check(_batch(9, 4200), BATCH_LEVELS, algorithm)
