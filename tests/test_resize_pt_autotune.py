"""resize_pt's lanczos path issues few conv shapes and stays bit-identical.

cudnn.benchmark autotunes every new conv shape (tens of ms each), and the lanczos
path issued a new one for every kernel length and input size that the random OTF
scales produced. The kernel is now zero-padded to a radius of 7, 15, 31, ... and
the padded input is extended with zeros to a size bucket and cropped after the conv.

The reference is traiNNer/data/degradations.py at REFERENCE_COMMIT, the last
commit before the change, loaded from git as a separate module. With
cudnn.benchmark on, cuDNN picks each conv algorithm by timing, and for inputs
above about 280 px the reference itself then differs by 1-2 ulp between
processes. Equality with benchmark on is therefore checked at the OTF's 256 px
input, and at varying sizes with benchmark off, where the choice is deterministic.
"""

from __future__ import annotations

import importlib.util
import random
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor
from traiNNer.data import degradations
from traiNNer.data.degradations import (
    _lanczos,
    _lanczos_kernel,
    _ramp,
    _size_bucket,
)

REFERENCE_COMMIT = "b6812173"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODES = ["bilinear", "bicubic", "nearest-exact", "lanczos"]
MODE_WEIGHTS = [0.3, 0.3, 0.1, 0.3]  # the owner's OTF config

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# (h, w, target h, target w) where the padded radius is capped at size - 1,
# plus odd and non-square sizes.
SMALL_CASES = [(12, 12, 4, 4), (20, 17, 6, 9), (33, 40, 11, 13), (31, 97, 30, 41)]


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    try:
        source = subprocess.run(
            ["git", "show", f"{REFERENCE_COMMIT}:traiNNer/data/degradations.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"reference commit {REFERENCE_COMMIT} is not available: {error}")
    path = tmp_path_factory.mktemp("reference") / "degradations_reference.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("degradations_reference", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image(batch: int, height: int, width: int, seed: int, device: str) -> Tensor:
    """Noise plus a smooth gradient and a hard edge, in [0, 1]."""
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.rand(batch, 3, height, width, device=device, generator=generator)
    ramp = torch.linspace(0.0, 1.0, width, device=device).expand(height, width)
    edge = (torch.arange(height, device=device) > height // 2).float()[:, None]
    return (0.5 * noise + 0.3 * ramp + 0.2 * edge).clamp(0, 1)


def _otf_sizes(count: int, seed: int) -> list[tuple[int, int, int, int]]:
    """Input and target sizes of the second and final OTF resizes (scale 2, GT 256:
    input 288 after a first resize in [0.4, 1.5], targets 144 * [0.6, 1.2] or 144),
    a fifth of them non-square."""
    rng = random.Random(seed)
    sizes = []
    for _ in range(count):
        h = round(288 * rng.uniform(0.4, 1.5))
        w = h if rng.random() < 0.8 else round(288 * rng.uniform(0.4, 1.5))
        target = int(144 * rng.uniform(0.6, 1.2)) if rng.random() < 0.5 else 144
        sizes.append((h, w, target, target))
    return sizes


def _record_conv_shapes(monkeypatch: pytest.MonkeyPatch) -> set[tuple[Any, ...]]:
    shapes: set[tuple[Any, ...]] = set()
    conv2d: Callable[..., Tensor] = torch.nn.functional.conv2d

    def recording_conv2d(
        inp: Tensor, weight: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        shapes.add((tuple(inp.shape), tuple(weight.shape)))
        return conv2d(inp, weight, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "conv2d", recording_conv2d)
    return shapes


@needs_cuda
@pytest.mark.parametrize("benchmark", [True, False])
def test_bit_identical_at_otf_size(
    reference: ModuleType, monkeypatch: pytest.MonkeyPatch, benchmark: bool
) -> None:
    """100 random (scale, mode) draws on the OTF's 16 x 3 x 256 x 256 batch."""
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", benchmark)
    image = _image(16, 256, 256, seed=0, device="cuda")
    rng = random.Random(4321)
    downscales = 0
    for draw in range(100):
        scale = rng.uniform(0.4, 1.5)
        mode = rng.choices(MODES, MODE_WEIGHTS)[0]
        downscales += mode == "lanczos" and round(256 * scale) < 256
        expected = reference.resize_pt(image, mode, scale_factor=scale)
        actual = degradations.resize_pt(image, mode, scale_factor=scale)
        assert torch.equal(actual, expected), f"draw {draw}: {mode} {scale}"
    assert downscales >= 10


@needs_cuda
def test_bit_identical_at_varying_sizes(
    reference: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    for i, (h, w, th, tw) in enumerate(SMALL_CASES + _otf_sizes(40, seed=5)):
        image = _image(4, h, w, seed=i, device="cuda")
        expected = reference.resize_pt(image, "lanczos", size=(th, tw))
        actual = degradations.resize_pt(image, "lanczos", size=(th, tw))
        assert torch.equal(actual, expected), f"{h}x{w} -> {th}x{tw}"


def test_bit_identical_on_cpu(reference: ModuleType) -> None:
    for i, (h, w, th, tw) in enumerate(SMALL_CASES + _otf_sizes(6, seed=6)):
        image = _image(2, h, w, seed=i, device="cpu")
        expected = reference.resize_pt(image, "lanczos", size=(th, tw))
        actual = degradations.resize_pt(image, "lanczos", size=(th, tw))
        assert torch.equal(actual, expected), f"{h}x{w} -> {th}x{tw}"


def test_one_conv_shape_per_direction_at_fixed_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every downscale in [0.4, 1) of a 256 px input uses the same two convs."""
    shapes = _record_conv_shapes(monkeypatch)
    image = _image(1, 256, 256, seed=0, device="cpu")
    for scale in (0.4, 0.45, 0.55, 0.62, 0.7, 0.81, 0.9, 0.99):
        degradations.resize_pt(image, "lanczos", scale_factor=scale)
    assert len(shapes) == 2


def test_conv_shapes_bounded_at_otf_sizes(
    reference: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Square inputs of 115-432 px fall in 8 size buckets and need a radius of 7
    or 15, so at most 2 directions x 8 x 2 shapes."""
    sizes = [s for s in _otf_sizes(80, seed=7) if s[0] == s[1]]
    shapes = _record_conv_shapes(monkeypatch)
    for h, w, th, tw in sizes:
        degradations.resize_pt(torch.zeros(1, 1, h, w), "lanczos", size=(th, tw))
    new_shapes = set(shapes)
    shapes.clear()
    for h, w, th, tw in sizes:
        reference.resize_pt(torch.zeros(1, 1, h, w), "lanczos", size=(th, tw))
    assert len(new_shapes) <= 32, (len(new_shapes), len(shapes))
    assert len(shapes) > 2 * len(new_shapes), (len(new_shapes), len(shapes))
    assert {weight[2] * weight[3] for _, weight in new_shapes} <= {15, 31}


@pytest.mark.parametrize("size", [6, 12, 16, 64, 255, 256, 288, 432, 1024])
def test_lanczos_kernel_is_the_original_zero_padded(size: int) -> None:
    for dsize in range(max(1, size // 6), size):
        original = _lanczos(_ramp(dsize / size, 3), 3)
        radius = (original.shape[0] - 1) // 2
        if radius >= size:
            continue  # reflect padding fails for the original as well
        kernel, conv_radius = _lanczos_kernel(dsize, size, 3)
        assert kernel.shape[0] == 2 * conv_radius + 1
        assert conv_radius in {7, 15, 31, 63, 127, size - 1, radius}
        assert radius <= conv_radius < size
        extra = conv_radius - radius
        assert torch.equal(kernel[extra : extra + original.shape[0]], original)
        assert not kernel[:extra].any()
        assert not kernel[extra + original.shape[0] :].any()


def test_size_bucket() -> None:
    for size in range(1, 2049):
        bucket = _size_bucket(size)
        assert bucket >= size
        assert bucket - size < max(32, size // 4)
    buckets = {_size_bucket(size) for size in range(256, 512)}
    assert buckets == {256, 320, 384, 448, 512}
