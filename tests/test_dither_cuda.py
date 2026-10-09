"""Bit-exactness of the CUDA dithering kernels against chainner_ext."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor
from traiNNer.data.dither_cuda import (
    DIFFUSION_ALGORITHMS,
    dither_cuda_available,
    error_diffusion_dither_cuda,
    palette_dither_cuda,
    palette_error_diffusion_dither_cuda,
    palette_quantization_cuda,
    palette_quantize_cuda,
    palette_riemersma_dither_cuda,
    riemersma_dither_cuda,
)
from traiNNer.data.otf_degradations import apply_dithering_palette, apply_per_image

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
PALETTE_SIZES = [(64, 64), (128, 128), (257, 131), (256, 64)]
PALETTE_COLORS = (2, 16, 128, 256)
PALETTE_BATCH_COLORS = (2, 16, 128, 256, 7, 64, 200, 3)
PALETTE_HISTORIES = (2, 8)
PALETTE_DECAYS = (0.1, 0.5)


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


# ── Palette mode ───────────────────────────────────────────────────────────────


def _assert_bits_equal(actual: Tensor, expected: Tensor, label: str) -> None:
    got = actual.cpu().numpy()
    want = expected.cpu().numpy()
    assert got.shape == want.shape and got.dtype == want.dtype, label
    mismatched = got.view(np.uint32) != want.view(np.uint32)
    if mismatched.any():
        diff = float(np.abs(got.astype(np.float64) - want).max())
        pytest.fail(
            f"{label}: {int(mismatched.sum())} values differ, max abs diff {diff}"
        )


def _check_palette(
    images: np.ndarray,
    dither: str,
    num_colors: int,
    history: int = 12,
    decay_ratio: float = 0.5,
) -> None:
    """The GPU batch against the per-image CPU path on the same tensor."""
    x = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).cuda()
    before = x.clone()
    actual = palette_dither_cuda(x, dither, num_colors, history, decay_ratio)
    assert torch.equal(x, before), "the input tensor was modified"
    expected = apply_per_image(
        x,
        lambda img: apply_dithering_palette(
            img, dither, num_colors, 4, history, decay_ratio
        ),
    )
    _assert_bits_equal(
        actual,
        expected,
        f"{dither} {num_colors} colors history {history} decay {decay_ratio} "
        f"batch {images.shape}",
    )


def _palette(size: int, channels: int, seed: int) -> np.ndarray:
    """A (P, C) palette of random colors; the last two repeat earlier ones."""
    colors = np.random.default_rng(seed).random((size, channels), dtype=np.float32)
    if size > 2:
        colors[-2:] = colors[:2]
    return colors


def _reference_palette(
    image: np.ndarray, palette: np.ndarray, dither: str, **kwargs: float
) -> np.ndarray:
    quant = chainner_ext.PaletteQuantization(palette[None])
    if dither == "riemersma":
        result = chainner_ext.riemersma_dither(
            image, quant, int(kwargs["history"]), float(kwargs["decay_ratio"])
        )
    elif dither == "quantize":
        result = chainner_ext.quantize(image, quant)
    else:
        result = chainner_ext.error_diffusion_dither(
            image, quant, ALGORITHM_ENUM[dither]
        )
    return np.asarray(result, dtype=np.float32).reshape(image.shape)


def _check_palettes(
    images: np.ndarray, palettes: list[np.ndarray], dither: str, **kwargs: float
) -> None:
    """The palette kernels on explicit palettes against chainner_ext per image."""
    x = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).cuda()
    uploaded, counts = palette_quantization_cuda(palettes, x.device)
    if dither == "riemersma":
        out = palette_riemersma_dither_cuda(
            x, uploaded, counts, int(kwargs["history"]), float(kwargs["decay_ratio"])
        )
    elif dither == "quantize":
        out = palette_quantize_cuda(x, uploaded, counts)
    else:
        out = palette_error_diffusion_dither_cuda(x, uploaded, counts, dither)
    actual = out.cpu().numpy().transpose(0, 2, 3, 1)
    for i, (image, palette) in enumerate(zip(images, palettes, strict=True)):
        expected = _reference_palette(
            np.ascontiguousarray(image), palette, dither, **kwargs
        )
        _assert_bits_equal(
            torch.from_numpy(np.ascontiguousarray(actual[i])),
            torch.from_numpy(expected),
            f"{dither} {kwargs} image {i} {image.shape} palette {len(palette)}",
        )


@pytest.mark.parametrize("dither", [*DIFFUSION_ALGORITHMS, "quantize"])
@pytest.mark.parametrize(("height", "width"), PALETTE_SIZES)
def test_palette_bit_exact(height: int, width: int, dither: str) -> None:
    images = _batch(height, width)
    _check_palette(images, dither, 256)
    for num_colors in PALETTE_COLORS[:-1]:
        _check_palette(images[:2], dither, num_colors)
    _check_palette(images[1:2], dither, 16)


@pytest.mark.parametrize("decay_ratio", PALETTE_DECAYS)
@pytest.mark.parametrize("history", PALETTE_HISTORIES)
@pytest.mark.parametrize(("height", "width"), PALETTE_SIZES)
def test_palette_riemersma_bit_exact(
    height: int, width: int, history: int, decay_ratio: float
) -> None:
    images = _batch(height, width)
    _check_palette(images, "riemersma", 256, history, decay_ratio)
    for num_colors in PALETTE_COLORS[:-1]:
        _check_palette(images[:2], "riemersma", num_colors, history, decay_ratio)
    _check_palette(images[1:2], "riemersma", 16, history, decay_ratio)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("dither", [*DIFFUSION_ALGORITHMS, "riemersma", "quantize"])
def test_palette_mixed_counts_bit_exact(dither: str, channels: int) -> None:
    """Per-image palettes of 2 to 256 colors, with repeats and NaN pixels."""
    images = _batch(61, 67, channels)
    images[3, 5:9, 7:11] = np.nan
    palettes = [
        _palette(size, channels, seed=i) for i, size in enumerate(PALETTE_BATCH_COLORS)
    ]
    _check_palettes(images, palettes, dither, history=12, decay_ratio=0.3)
    _check_palettes(images[:1], palettes[:1], dither, history=12, decay_ratio=0.3)


@pytest.mark.parametrize("channels", [1, 4])
@pytest.mark.parametrize(
    "dither", ["floydsteinberg", "jarvisjudiceninke", "riemersma", "quantize"]
)
def test_palette_channel_counts_bit_exact(dither: str, channels: int) -> None:
    """Gray takes a 1-channel palette; RGBA dithers the colors, alpha passes."""
    _check_palette(_batch(61, 67, channels), dither, 64, history=12, decay_ratio=0.3)


def test_palette_flat_images_pass_through() -> None:
    """Images without a usable palette (one color) come back unchanged."""
    images = _batch(32, 32)
    images[2] = 0.5
    images[5] = 0.0
    _check_palette(images, "floydsteinberg", 16)
    _check_palette(images[2:3], "floydsteinberg", 16)


def test_palette_tie_takes_the_lower_key_color() -> None:
    """A pixel midway between two colors takes the first in key order."""
    low = [0.25, 0.5, 0.75]
    high = [0.75, 0.5, 0.25]
    palettes = [np.array([high, low], np.float32), np.array([low, high], np.float32)]
    x = torch.full((2, 3, 4, 4), 0.5, device="cuda")
    uploaded, counts = palette_quantization_cuda(palettes, x.device)
    out = palette_quantize_cuda(x, uploaded, counts)
    assert out[:, :, 0, 0].tolist() == [low, low]
    reference = chainner_ext.quantize(
        np.full((4, 4, 3), 0.5, np.float32),
        chainner_ext.PaletteQuantization(palettes[0][None]),
    )
    assert np.asarray(reference)[0, 0].tolist() == low


@pytest.mark.parametrize("history", [33, 5000])
def test_palette_riemersma_long_history_bit_exact(history: int) -> None:
    """Histories past one per lane: the shared-memory and global-scratch rings."""
    _check_palette(_batch(17, 19), "riemersma", 32, history, 0.5)


@pytest.mark.parametrize("algorithm", ["floydsteinberg", "stucki"])
def test_palette_error_diffusion_wide_image_bit_exact(algorithm: str) -> None:
    """Rows wider than the shared-memory ring use the global-scratch ring."""
    _check_palette(_batch(9, 4200), algorithm, 64)
