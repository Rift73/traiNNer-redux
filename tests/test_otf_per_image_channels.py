"""The per-image CPU degradations keep the image's shape through apply_per_image
for 1, 3 and 4 channels.

Results that were already correct (3 channels; 4 channels for uniform dithering
and translucent WebP) must equal REFERENCE_COMMIT, the last commit before the
fix, loaded from git as a separate module.
"""

from __future__ import annotations

import importlib.util
import subprocess
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
from traiNNer.data import otf_degradations
from traiNNer.data.otf_degradations import (
    _extract_palette,
    apply_per_image,
    apply_shift,
)

pytest.importorskip("chainner_ext")

REFERENCE_COMMIT = "b6812173"
REPO_ROOT = Path(__file__).resolve().parent.parent

UNIFORM_TYPES = ("floydsteinberg", "stucki", "order", "riemersma", "quantize")
PALETTE_TYPES = ("floydsteinberg", "riemersma", "quantize")

# name -> (function, keyword arguments)
CASES: dict[str, tuple[str, dict[str, Any]]] = {
    **{
        f"uniform_{kind}": (
            "apply_dithering",
            {"dithering_type": kind, "quantize_ch": 4},
        )
        for kind in UNIFORM_TYPES
    },
    **{
        f"palette_{kind}": (
            "apply_dithering_palette",
            {"dithering_type": kind, "num_colors": 8},
        )
        for kind in PALETTE_TYPES
    },
    "webp": ("compress_webp", {"quality": 80}),
}
DITHER_CASES = sorted(case for case in CASES if case != "webp")
UNCHANGED = [
    *((case, 3) for case in sorted(CASES)),
    *((f"uniform_{kind}", 4) for kind in UNIFORM_TYPES),
    ("webp", 4),
]


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    try:
        source = subprocess.run(
            ["git", "show", f"{REFERENCE_COMMIT}:traiNNer/data/otf_degradations.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"reference commit {REFERENCE_COMMIT} is not available: {error}")
    path = tmp_path_factory.mktemp("reference") / "otf_degradations_reference.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("otf_degradations_reference", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _batch(channels: int, height: int = 24, width: int = 20) -> torch.Tensor:
    """Noise over a horizontal ramp, BCHW float32 in [0, 1]."""
    generator = torch.Generator().manual_seed(channels)
    noise = torch.rand(2, channels, height, width, generator=generator)
    ramp = torch.linspace(0.0, 1.0, width).expand(height, width)
    return (0.5 * noise + 0.5 * ramp).clamp(0, 1)


def _run(module: ModuleType, case: str, batch: torch.Tensor) -> torch.Tensor:
    name, kwargs = CASES[case]
    fn = partial(getattr(module, name), **kwargs)
    if name == "compress_webp":
        return module.apply_per_image_parallel(batch, fn)
    return module.apply_per_image(batch, fn)


@pytest.mark.parametrize(("case", "channels"), UNCHANGED)
def test_correct_results_unchanged(
    reference: ModuleType, case: str, channels: int
) -> None:
    batch = _batch(channels)
    assert torch.equal(
        _run(otf_degradations, case, batch), _run(reference, case, batch)
    )


@pytest.mark.parametrize("case", DITHER_CASES)
def test_one_channel_dithers_like_each_gray_rgb_channel(case: str) -> None:
    """Gray dithered as one channel equals every channel of the same gray
    dithered as RGB (same levels, or the same median-cut palette)."""
    gray = _batch(1)
    out = _run(otf_degradations, case, gray)
    rgb_out = _run(otf_degradations, case, gray.expand(-1, 3, -1, -1).contiguous())
    assert out.shape == gray.shape
    assert torch.equal(out.expand_as(rgb_out), rgb_out)


def test_one_channel_webp_round_trips() -> None:
    gray = _batch(1)
    out = _run(otf_degradations, "webp", gray)
    rgb_out = _run(otf_degradations, "webp", gray.expand(-1, 3, -1, -1).contiguous())
    assert out.shape == gray.shape
    # WebP codes gray as RGB; the gray result is its luma.
    assert (out - rgb_out.mean(dim=1, keepdim=True)).abs().max() <= 2 / 255


@pytest.mark.parametrize("kind", PALETTE_TYPES)
def test_four_channel_palette_dithers_color_and_keeps_alpha(kind: str) -> None:
    rgba = _batch(4)
    out = _run(otf_degradations, f"palette_{kind}", rgba)
    assert out.shape == rgba.shape
    assert torch.equal(
        out[:, :3], _run(otf_degradations, f"palette_{kind}", rgba[:, :3])
    )
    assert torch.equal(out[:, 3], rgba[:, 3])


def test_webp_keeps_an_opaque_alpha() -> None:
    """WebP drops a fully opaque alpha channel; the result keeps all four."""
    rgba = _batch(4)
    rgba[:, 3] = 1.0
    out = _run(otf_degradations, "webp", rgba)
    assert out.shape == rgba.shape
    assert torch.equal(out[:, 3], rgba[:, 3])


@pytest.mark.parametrize("kind", UNIFORM_TYPES)
def test_single_row_image(kind: str) -> None:
    """np.squeeze used to drop the height of a 1-pixel-high image."""
    row = _batch(3, height=1)
    out = _run(otf_degradations, f"uniform_{kind}", row)
    assert out.shape == row.shape
    levels = out * 3  # quantize_ch 4: 0, 1/3, 2/3, 1
    assert (levels - levels.round()).abs().max() < 1e-5


def test_gray_palette_has_distinct_levels() -> None:
    """Pillow returns RGB triplets for a gray source; the palette takes one value
    per entry (not the first N values of the triplets)."""
    ramp = np.linspace(0.0, 1.0, 24 * 20, dtype=np.float32).reshape(24, 20, 1)
    palette = _extract_palette(ramp, 8)
    assert palette is not None
    assert palette.shape == (1, 8, 1)
    assert np.unique(palette).size == 8
    flat = _extract_palette(ramp[:, :, 0], 8)
    assert flat is not None
    assert np.array_equal(palette, flat)
    rgb = _extract_palette(np.repeat(ramp, 3, axis=2), 8)
    assert rgb is not None
    assert np.array_equal(palette[..., 0], rgb[..., 0])


def test_shift_leaves_gray_unchanged() -> None:
    gray = _batch(1)
    amounts = [[[1, 3], [1, 3]]] * 3
    out = apply_per_image(gray, partial(apply_shift, shift_type="rgb", amounts=amounts))
    assert torch.equal(out, gray)
