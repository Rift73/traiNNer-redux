"""Sync-free OTF noise and DiffJPEG, bit-exact against the original implementations.

The _ref_* functions copy the arithmetic of the implementations these replace
(per-image torch.unique and per-image quality loops with host syncs) op for op.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import Tensor
from torchvision.transforms.functional import rgb_to_grayscale
from traiNNer.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from traiNNer.utils.diffjpeg import CompressJpeg, DiffJPEG

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CLIP_ROUNDS = [(True, False), (False, True), (True, True), (False, False)]


# ----------------------------- references ----------------------------- #
def _ref_generate_gaussian_noise_pt(
    img: Tensor, sigma: float | Tensor = 10, gray_noise: float | Tensor = 0
) -> Tensor:
    b, _, h, w = img.size()
    if not isinstance(sigma, float | int):
        sigma = sigma.view(img.size(0), 1, 1, 1)
    if isinstance(gray_noise, float | int):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    noise_gray = None
    if cal_gray_noise:
        noise_gray = (
            torch.randn(*img.size()[2:4], dtype=img.dtype, device=img.device)
            * sigma
            / 255.0
        )
        noise_gray = noise_gray.view(b, 1, h, w)

    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.0

    if cal_gray_noise and noise_gray is not None:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise


def _ref_generate_poisson_noise_pt(
    img: Tensor, scale: float | Tensor = 1.0, gray_noise: float | Tensor = 0
) -> Tensor:
    b, _, h, w = img.size()
    if isinstance(gray_noise, float | int):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    noise_gray = None
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.0
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.0
    vals_list = [len(torch.unique(img[i, :, :, :])) for i in range(b)]
    vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
    vals = img.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img * vals) / vals
    noise = out - img
    if cal_gray_noise and noise_gray is not None:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, float | int):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale


def _clip_round(out: Tensor, clip: bool, rounds: bool) -> Tensor:
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0
    return out


def _ref_random_add_gaussian_noise_pt(
    img: Tensor,
    sigma_range: tuple[float, float],
    gray_prob: float,
    clip: bool,
    rounds: bool,
) -> Tensor:
    sigma = (
        torch.rand(img.size(0), dtype=img.dtype, device=img.device)
        * (sigma_range[1] - sigma_range[0])
        + sigma_range[0]
    )
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    noise = _ref_generate_gaussian_noise_pt(img, sigma, gray_noise)
    return _clip_round(img + noise, clip, rounds)


def _ref_random_add_poisson_noise_pt(
    img: Tensor,
    scale_range: tuple[float, float],
    gray_prob: float,
    clip: bool,
    rounds: bool,
) -> Tensor:
    scale = (
        torch.rand(img.size(0), dtype=img.dtype, device=img.device)
        * (scale_range[1] - scale_range[0])
        + scale_range[0]
    )
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    noise = _ref_generate_poisson_noise_pt(img, scale, gray_noise)
    return _clip_round(img + noise, clip, rounds)


def _ref_quality_to_factor(quality: Tensor) -> Tensor:
    if quality < 50:
        quality = 5000.0 / quality
    else:
        quality = 200.0 - quality * 2
    return quality / 100.0


def _ref_diffjpeg(jpeger: DiffJPEG, x: Tensor, quality: Tensor) -> Tensor:
    factor = quality.clone()
    for i in range(factor.size(0)):
        factor[i] = _ref_quality_to_factor(factor[i])
    h, w = x.size()[-2:]
    h_pad = 16 - h % 16 if h % 16 else 0
    w_pad = 16 - w % 16 if w % 16 else 0
    x = torch.nn.functional.pad(x, (0, w_pad, 0, h_pad), mode="constant", value=0)
    y, cb, cr = jpeger.compress(x, factor=factor)
    recovered = jpeger.decompress(y, cb, cr, (h + h_pad), (w + w_pad), factor=factor)
    return recovered[:, :, 0:h, 0:w]


# ------------------------------- helpers ------------------------------- #
def _assert_bitwise(actual: Tensor, expected: Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape
    same = actual.view(torch.int32) == expected.view(torch.int32)
    assert bool(same.all()), (
        f"{int((~same).sum())} of {same.numel()} elements differ bitwise, "
        f"max abs diff {float((actual - expected).abs().nan_to_num(0).max())}"
    )


def _images(seed: int, h: int = 48, w: int = 80) -> Tensor:
    """Eight images with different value statistics (b, 3, h, w)."""
    g = torch.Generator().manual_seed(seed)
    shape = (3, h, w)
    levels = (5, 64, 65, 2)
    imgs = [
        torch.rand(shape, generator=g),
        torch.rand(shape, generator=g) * 1.4 - 0.2,  # outside [0, 1]
        torch.full(shape, 0.37),  # one level
    ]
    for n in levels:
        k = torch.randint(0, n, shape, generator=g).float()
        imgs.append(k * (1.0 / (n - 1)))
    # +0.0 and -0.0 next to small positives
    r = torch.rand(shape, generator=g)
    signed = r * 0.01
    signed[r < 0.4] = 0.0
    signed[r < 0.2] = -0.0
    imgs.append(signed)
    return torch.stack(imgs).cuda()


def _level_images(channels: int = 3, side: int = 16) -> Tensor:
    """256 images; image n-1 holds exactly n distinct 8-bit levels."""
    g = torch.Generator().manual_seed(7)
    pixels = channels * side * side
    imgs = []
    for n in range(1, 257):
        k = torch.from_numpy(np.round(np.linspace(0, 255, n))).float()
        order = torch.randperm(pixels, generator=g) % n
        imgs.append((k[order] / 255.0).view(channels, side, side))
    return torch.stack(imgs).cuda()


def _count_syncs(fn: Callable[[], object]) -> int:
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    # Count sync reports only, not the one-time "prototype feature" notice.
    return sum("called a synchronizing" in str(w.message) for w in caught)


def _seeded(seed: int, fn: Callable[[], Tensor]) -> tuple[Tensor, Tensor]:
    """Output of fn after seeding, and the CUDA generator state after it."""
    torch.manual_seed(seed)
    out = fn()
    return out, torch.cuda.get_rng_state()


# -------------------------------- noise -------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("gray_prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(("clip", "rounds"), CLIP_ROUNDS)
def test_poisson_bitwise(seed: int, gray_prob: float, clip: bool, rounds: bool) -> None:
    img = _images(seed)
    args = ((0.05, 3.0), gray_prob, clip, rounds)
    ref, ref_rng = _seeded(seed, lambda: _ref_random_add_poisson_noise_pt(img, *args))
    new, new_rng = _seeded(seed, lambda: random_add_poisson_noise_pt(img, *args))
    _assert_bitwise(new, ref)
    assert torch.equal(new_rng, ref_rng), "random stream position changed"


@pytest.mark.parametrize("gray_prob", [0.0, 1.0])
def test_poisson_every_level_count(gray_prob: float) -> None:
    """Covers every distinct-value count an 8-bit image can have (1..256)."""
    img = _level_images()
    args = ((0.5, 2.0), gray_prob, True, False)
    ref, _ = _seeded(3, lambda: _ref_random_add_poisson_noise_pt(img, *args))
    new, _ = _seeded(3, lambda: random_add_poisson_noise_pt(img, *args))
    _assert_bitwise(new, ref)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("gray_prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(("clip", "rounds"), CLIP_ROUNDS)
def test_poisson_zero_scale_skip(
    seed: int, gray_prob: float, clip: bool, rounds: bool
) -> None:
    """scale_range [0, 0] (skipped when gray_prob is 0) equals the full path."""
    img = _images(seed)
    args = ((0.0, 0.0), gray_prob, clip, rounds)
    ref, _ = _seeded(seed, lambda: _ref_random_add_poisson_noise_pt(img, *args))
    new, _ = _seeded(seed, lambda: random_add_poisson_noise_pt(img, *args))
    _assert_bitwise(new, ref)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("gray_prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(("clip", "rounds"), CLIP_ROUNDS)
def test_gaussian_bitwise(
    seed: int, gray_prob: float, clip: bool, rounds: bool
) -> None:
    img = _images(seed)
    args = ((1.0, 30.0), gray_prob, clip, rounds)
    ref, ref_rng = _seeded(seed, lambda: _ref_random_add_gaussian_noise_pt(img, *args))
    new, new_rng = _seeded(seed, lambda: random_add_gaussian_noise_pt(img, *args))
    _assert_bitwise(new, ref)
    assert torch.equal(new_rng, ref_rng), "random stream position changed"


# ------------------------------- DiffJPEG ------------------------------- #
@pytest.fixture(scope="module")
def jpeger() -> DiffJPEG:
    return DiffJPEG(differentiable=False).cuda()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(("h", "w"), [(64, 64), (72, 100)])
def test_diffjpeg_bitwise_random_quality(
    jpeger: DiffJPEG, seed: int, h: int, w: int
) -> None:
    torch.manual_seed(seed)
    x = torch.rand(8, 3, h, w, device="cuda")
    quality = torch.empty(8, device="cuda").uniform_(1, 99)
    original = quality.clone()
    _assert_bitwise(jpeger(x, quality=quality), _ref_diffjpeg(jpeger, x, quality))
    assert torch.equal(quality, original), "quality tensor was modified"


def test_diffjpeg_bitwise_boundary_quality(jpeger: DiffJPEG) -> None:
    torch.manual_seed(4)
    x = torch.rand(8, 3, 64, 64, device="cuda")
    quality = torch.tensor(
        [1.0, 10.0, 49.999, 50.0, 50.001, 75.0, 98.5, 99.99999], device="cuda"
    )
    _assert_bitwise(jpeger(x, quality=quality), _ref_diffjpeg(jpeger, x, quality))


@torch.no_grad()
def test_diffjpeg_quality_100_near_lossless(jpeger: DiffJPEG) -> None:
    """Quality 100 is finite and matches the pipeline with quantisation removed."""
    torch.manual_seed(5)
    x = torch.rand(4, 3, 64, 72, device="cuda")
    unquantised = DiffJPEG(differentiable=False).cuda()
    unquantised.compress = CompressJpeg(rounding=lambda t: t).cuda()
    expected = unquantised(x, quality=50.0)  # factor 1, no rounding
    for quality in (100.0, torch.full((4,), 100.0, device="cuda")):
        out = jpeger(x, quality=quality)
        assert bool(torch.isfinite(out).all())
        assert float((out - expected).abs().max()) < 1e-4
    gray = x[:, :1].expand(4, 3, 64, 72)  # no chroma, so no subsampling loss
    out = jpeger(gray, quality=torch.full((4,), 100.0, device="cuda"))
    assert float((out - gray).abs().max()) < 1e-4


# ------------------------------ host syncs ------------------------------ #
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("poisson gray 0", 0),
        ("poisson gray 1", 0),
        ("poisson gray 0.5", 1),
        ("poisson scale [0, 0]", 0),
        ("gaussian gray 0", 0),
        ("gaussian gray 1", 0),
        ("gaussian gray 0.5", 1),
        ("diffjpeg", 0),
    ],
)
def test_host_syncs(jpeger: DiffJPEG, name: str, expected: int) -> None:
    """A gray_prob strictly between 0 and 1 keeps one sync: the gray switch."""
    img = _images(0)
    quality = torch.empty(8, device="cuda").uniform_(1, 99)
    calls: dict[str, Callable[[], object]] = {
        "poisson gray 0": lambda: random_add_poisson_noise_pt(img, (0.05, 3.0), 0.0),
        "poisson gray 1": lambda: random_add_poisson_noise_pt(img, (0.05, 3.0), 1.0),
        "poisson gray 0.5": lambda: random_add_poisson_noise_pt(img, (0.05, 3.0), 0.5),
        "poisson scale [0, 0]": lambda: random_add_poisson_noise_pt(img, (0, 0), 0.0),
        "gaussian gray 0": lambda: random_add_gaussian_noise_pt(img, (1, 30), 0.0),
        "gaussian gray 1": lambda: random_add_gaussian_noise_pt(img, (1, 30), 1.0),
        "gaussian gray 0.5": lambda: random_add_gaussian_noise_pt(img, (1, 30), 0.5),
        "diffjpeg": lambda: jpeger(img, quality=quality),
    }
    calls[name]()  # warm-up
    assert _count_syncs(calls[name]) == expected
