"""channel_shift_pt draws its shifts from the project RNG, so they follow the run
seed: the same seed repeats the shifts and the output, another seed changes them.

channel_shift_pt is plain torch, so this runs on the CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from traiNNer.data import gpu_degradations
from traiNNer.utils.rng import RNG

# The unpatched helper, so every recording wrapper calls the real one.
_SHIFT_CHANNEL = vars(gpu_degradations)["_shift_channel"]

# [[x_lo, x_hi], [y_lo, y_hi]] per channel; cmyk uses all four.
PIXEL_AMOUNTS = [
    [[-3, 3], [-2, 2]],
    [[0, 4], [-4, 0]],
    [[-1, 1], [1, 3]],
    [[-2, 2], [-3, 3]],
]
PERCENT_AMOUNTS = [[[-12, 12], [-12, 12]]] * 4

Shifts = list[tuple[int, int]]


def _seeded_shift(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    shift_type: str,
    amounts: list[list[list[int]]],
    percent: bool,
) -> tuple[Shifts, torch.Tensor]:
    """channel_shift_pt with the project RNG freshly seeded.

    Returns the (dx, dy) applied to each channel, in order, and the output.
    """
    shifts: Shifts = []

    def recording_shift_channel(
        channel: torch.Tensor, dx: int, dy: int, fill_value: float = 1.0
    ) -> torch.Tensor:
        shifts.append((dx, dy))
        return _SHIFT_CHANNEL(channel, dx, dy, fill_value)

    monkeypatch.setattr(gpu_degradations, "_shift_channel", recording_shift_channel)
    monkeypatch.setattr(RNG, "_rng", np.random.default_rng(seed))
    image = torch.rand(2, 3, 40, 48, generator=torch.Generator().manual_seed(0))
    out = gpu_degradations.channel_shift_pt(image, shift_type, amounts, percent)
    return shifts, out


@pytest.mark.parametrize("percent", [False, True])
@pytest.mark.parametrize("shift_type", ["rgb", "yuv", "cmyk"])
def test_same_seed_gives_same_shifts(
    monkeypatch: pytest.MonkeyPatch, shift_type: str, percent: bool
) -> None:
    amounts = PERCENT_AMOUNTS if percent else PIXEL_AMOUNTS
    first, first_out = _seeded_shift(monkeypatch, 123, shift_type, amounts, percent)
    again, again_out = _seeded_shift(monkeypatch, 123, shift_type, amounts, percent)
    other, _ = _seeded_shift(monkeypatch, 7, shift_type, amounts, percent)

    assert len(first) == (4 if shift_type == "cmyk" else 3)
    assert any(shift != (0, 0) for shift in first)
    assert first == again
    assert torch.equal(first_out, again_out)
    assert other != first


def test_pixel_shifts_cover_the_inclusive_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[lo, hi] draws every integer from lo to hi inclusive, as the previous
    np.random.randint(lo, hi + 1) did; a reversed range is swapped and [v, v]
    gives v without a draw."""
    amounts = [[[-2, 2], [3, 0]], [[1, 1], [0, 0]], [[0, 0], [-1, -1]]]
    seen_x: set[int] = set()
    seen_y: set[int] = set()
    for seed in range(60):
        (dx, dy), second, third = _seeded_shift(
            monkeypatch, seed, "rgb", amounts, percent=False
        )[0]
        seen_x.add(dx)
        seen_y.add(dy)
        assert second == (1, 0)
        assert third == (0, -1)
    assert seen_x == {-2, -1, 0, 1, 2}
    assert seen_y == {0, 1, 2, 3}
