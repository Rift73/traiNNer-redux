"""The GPU degradations stay bit-identical after the host-sync removal, and sync-free.

The reference is traiNNer/data/gpu_degradations.py at REFERENCE_COMMIT, the last
commit before the change, loaded from git as a separate module. Both modules get
the same seeded parameters and RNG state, and their outputs must be torch.equal.

Exception: the reference NTSC FIR cache keyed filters on round(kernel.sum() * 1e8),
and every kernel is normalised to sum 1, so filters with the same tap count can
share a key. That happens for the VHS filters (61 taps) and for Y/I/Q once the
input exceeds 754x480 (scale 2). Those paths are compared with the reference
cache disabled, i.e. with the filters the reference meant to apply.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
from traiNNer.data import gpu_degradations
from traiNNer.utils.rng import RNG

REFERENCE_COMMIT = "0ed482ff"
REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

Params = tuple[tuple[Any, ...], dict[str, Any]]
Sampler = Callable[[np.random.Generator], Params]


@pytest.fixture(autouse=True, scope="module")
def _project_rng() -> None:
    # channel_shift_pt draws its shifts from the project RNG.
    with contextlib.suppress(RuntimeError):  # already initialised
        RNG.init_rng(0)


def _channel_shift_params(r: np.random.Generator) -> Params:
    """Shifts drawn here and passed as [v, v] ranges.

    channel_shift_pt resolves a [v, v] range to v whatever the RNG state (the
    reference draws from numpy's global RNG, the current code from the project
    RNG), so both modules apply the same shifts.
    """
    shift_type = str(r.choice(["rgb", "yuv", "cmyk"]))
    percent = bool(r.integers(0, 2))
    channels = 4 if shift_type == "cmyk" else 3
    amounts = []
    for _ in range(channels):
        if percent:
            dx, dy = (float(v) for v in r.uniform(-3.0, 3.0, size=2))
        else:
            dx, dy = (int(v) for v in r.integers(-3, 4, size=2))
        amounts.append([[dx, dx], [dy, dy]])
    return (shift_type, amounts, percent), {}


def _ntsc_params(r: np.random.Generator) -> Params:
    """Broadcast-style NTSC draws covering every effect except VHS."""
    return (), {
        "noise": float(r.uniform(0.0, 0.1)),
        "luma_noise": float(r.choice([0.0, r.uniform(0.0, 0.15)])),
        "ghost_amplitude": float(r.choice([0.0, r.uniform(0.0, 0.5)])),
        "ghost_delay_us": float(r.uniform(0.5, 10.0)),
        "ghost_phase": float(r.uniform(0.0, 360.0)),
        "jitter": float(r.choice([0.0, r.uniform(0.0, 3.0)])),
        "edge_ringing": float(r.choice([0.0, r.uniform(0.0, 3.0)])),
        "tape_trailing": float(r.choice([0.0, r.uniform(0.0, 1.0)])),
        "intensity": float(r.choice([1.0, r.uniform(0.3, 1.0)])),
        "comb_mode": str(r.choice(["2sample", "1h"])),
    }


def _ntsc_vhs_params(r: np.random.Generator) -> Params:
    args, kwargs = _ntsc_params(r)
    kwargs.update(
        enable_vhs=True,
        vhs_luma_bw=float(r.uniform(1.5, 4.2)),
        color_under_bw=float(r.uniform(200.0, 600.0)),
    )
    return args, kwargs


# name -> (function, parameter sampler); ranges follow OTF_documentation.md.
CASES: dict[str, tuple[str, Sampler]] = {
    "ntsc": ("ntsc_composite_pt", _ntsc_params),
    "ntsc_vhs": ("ntsc_composite_pt", _ntsc_vhs_params),
    "rainbow": (
        "composite_rainbow_pt",
        lambda r: (
            (
                float(r.uniform(0.15, 0.35)),
                float(r.uniform(0.005, 0.15)),
                float(r.uniform(0.3, 1.0)),
                bool(r.integers(0, 2)),
                float(r.uniform(0.0, 2.0 * math.pi)),
            ),
            {},
        ),
    ),
    "lowpass": (
        "lowpass_filter_pt",
        lambda r: ((float(r.uniform(0.3, 0.8)), int(r.integers(1, 6))), {}),
    ),
    "overshoot": (
        "overshoot_pt",
        lambda r: (
            (
                float(r.uniform(0.5, 2.0)),
                float(r.uniform(0.2, 0.5)),
                int(r.integers(1, 4)),
            ),
            {},
        ),
    ),
    "interlace": (
        "interlace_pt",
        lambda r: ((int(r.integers(0, 6)), str(r.choice(["top", "bottom"]))), {}),
    ),
    "ghosting": (
        "temporal_ghosting_pt",
        lambda r: (
            (
                int(r.integers(0, 9)),
                int(r.integers(0, 3)),
                float(r.uniform(0.05, 0.25)),
            ),
            {},
        ),
    ),
    "scanline": (
        "scanline_pt",
        lambda r: ((float(r.uniform(0.1, 0.5)), bool(r.integers(0, 2))), {}),
    ),
    "channel_shift": ("channel_shift_pt", _channel_shift_params),
    "chroma_subsample": (
        "chroma_subsample_pt",
        lambda r: (
            (
                str(r.choice(["nearest", "linear", "cubic_catrom", "lanczos"])),
                str(r.choice(["nearest", "linear", "cubic_mitchell"])),
                str(r.choice(["4:2:0", "4:2:2", "4:1:1", "4:4:0", "4:1:3"])),
                r.choice([None, float(r.uniform(0.5, 2.0))]),
                str(r.choice(["601", "709", "2020", "240"])),
            ),
            {},
        ),
    ),
    "quantize": ("quantize_pt", lambda r: ((int(r.integers(2, 33)),), {})),
    "ordered_dither": (
        "ordered_dither_pt",
        lambda r: ((int(r.integers(2, 33)), int(r.choice([2, 3, 4, 8, 16]))), {}),
    ),
    "detail_mask": (
        "detail_mask_neo_pt",
        lambda r: (
            (),
            {
                "sigma": float(r.choice([1.0, 1.5])),
                "lines_brz": float(r.uniform(0.04, 0.12)),
            },
        ),
    ),
    "ycbcr_round_trip": (
        "ycbcr_to_rgb_pt",
        lambda r: ((str(r.choice(["601", "709", "2020", "240"])),), {}),
    ),
}


class _AlwaysMiss(dict):
    """Stands in for the reference FIR cache so every filter is designed fresh."""

    def __contains__(self, key: object) -> bool:
        return False


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    try:
        source = subprocess.run(
            ["git", "show", f"{REFERENCE_COMMIT}:traiNNer/data/gpu_degradations.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"reference commit {REFERENCE_COMMIT} is not available: {error}")
    path = tmp_path_factory.mktemp("reference") / "gpu_degradations_reference.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("gpu_degradations_reference", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image(batch: int, height: int, width: int, seed: int) -> torch.Tensor:
    """Noise plus a smooth gradient and a hard edge, in [0, 1]."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.rand(batch, 3, height, width, device="cuda", generator=generator)
    ramp = torch.linspace(0.0, 1.0, width, device="cuda").expand(height, width)
    edge = (torch.arange(height, device="cuda") > height // 2).float()[:, None]
    return (0.5 * noise + 0.3 * ramp + 0.2 * edge).clamp(0, 1)


def _call(
    module: ModuleType, name: str, image: torch.Tensor, params: Params, seed: int
) -> torch.Tensor:
    args, kwargs = params
    torch.manual_seed(seed)  # the NTSC noise and jitter draws
    if name == "ycbcr_to_rgb_pt":
        image = module.rgb_to_ycbcr_pt(image, *args)
    return getattr(module, name)(image, *args, **kwargs)


def _assert_matches(
    reference: ModuleType,
    case: str,
    image: torch.Tensor,
    draws: int,
    atol: float = 0.0,
) -> None:
    """Bit-identical to the reference, or within atol when atol is not 0."""
    name, sampler = CASES[case]
    rng = np.random.default_rng(sum(map(ord, case)))
    for draw in range(draws):
        params = sampler(rng)
        expected = _call(reference, name, image, params, seed=draw)
        actual = _call(gpu_degradations, name, image, params, seed=draw)
        if atol:
            difference = float((actual - expected).abs().max())
            assert difference <= atol, f"{case} draw {draw}: {params}: max |d| {difference}"
        else:
            assert torch.equal(actual, expected), f"{case} draw {draw}: {params}"


@pytest.mark.parametrize("case", sorted(set(CASES) - {"ntsc_vhs"}))
def test_bit_identical_to_reference(reference: ModuleType, case: str) -> None:
    _assert_matches(reference, case, _image(4, 96, 160, seed=1), draws=8)


@pytest.mark.parametrize("width", [160, 40, 6])
def test_rainbow_bit_identical_with_cudnn_benchmark(
    reference: ModuleType, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """Training runs with cudnn.benchmark on, which picks the conv algorithm by
    timing, separately for the padded and the unpadded kernel shape; the two
    choices can differ (once, in a 15-minute run, by an ulp), so the padded
    kernel must match within 1e-6 there, including widths that cap the padded
    radius at w // 2. Exact equality is test_bit_identical_to_reference's."""
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    _assert_matches(reference, "rainbow", _image(4, 96, width, seed=4), draws=12, atol=1e-6)


@pytest.mark.parametrize(
    ("case", "shape"),
    [("ntsc", (4, 96, 160)), ("ntsc", (1, 488, 96)), ("ntsc_vhs", (4, 96, 160))],
)
def test_ntsc_matches_uncached_reference(
    reference: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    shape: tuple[int, int, int],
) -> None:
    monkeypatch.setattr(reference, "_ntsc_filter_cache", _AlwaysMiss())
    _assert_matches(reference, case, _image(*shape, seed=2), draws=4)


@pytest.mark.parametrize("case", sorted(CASES))
def test_no_host_sync_after_warm_up(case: str) -> None:
    name, sampler = CASES[case]
    image = _image(4, 96, 160, seed=3)
    rng = np.random.default_rng(7)
    _call(gpu_degradations, name, image, sampler(rng), seed=0)  # fills the caches
    params = [sampler(rng) for _ in range(4)]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for seed, draw in enumerate(params, start=1):
            _call(gpu_degradations, name, image, draw, seed=seed)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
