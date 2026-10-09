import numpy as np
import pytest
import pyvips
import torch
from traiNNer.data.realesrgan_dataset import _reflect_pad_vips
from traiNNer.data.transforms import paired_random_crop, paired_random_crop_multi_gt


@pytest.mark.parametrize(
    ("source_height", "expected_reflected_rows"), ((325, 59), (375, 9))
)
def test_otf_reflect_padding_is_symmetric(
    source_height: int, expected_reflected_rows: int
) -> None:
    source = np.arange(source_height * 500 * 3, dtype=np.uint8).reshape(
        source_height, 500, 3
    )
    image = pyvips.Image.new_from_memory(source.data, 500, source_height, 3, "uchar")

    padded = _reflect_pad_vips(image, 416)
    pad_top = (416 - source_height) // 2
    final_crop_top = (416 - 384) // 2
    native_rows = min(pad_top + source_height, final_crop_top + 384) - max(
        pad_top, final_crop_top
    )

    assert (padded.height, padded.width) == (416, 500)
    np.testing.assert_array_equal(padded.getpoint(0, pad_top), source[0, 0])
    assert 384 - native_rows == expected_reflected_rows


def test_otf_final_crop_centers_only_padded_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lq = torch.arange(104 * 104).reshape(1, 1, 104, 104)
    gt = torch.arange(416 * 416).reshape(1, 1, 416, 416)
    monkeypatch.setattr("traiNNer.data.transforms.random.randint", lambda _a, b: b)

    cropped_gt, cropped_lq = paired_random_crop(gt, lq, 384, 4, center_height=True)
    cropped_gts, cropped_multi_lq = paired_random_crop_multi_gt(
        [gt, gt + 1], lq, 384, 4, center_height=True
    )

    assert torch.equal(cropped_lq, lq[:, :, 4:100, 8:104])
    assert torch.equal(cropped_gt, gt[:, :, 16:400, 32:416])
    assert torch.equal(cropped_multi_lq, cropped_lq)
    assert torch.equal(cropped_gts[0], cropped_gt)
    assert torch.equal(cropped_gts[1], cropped_gt + 1)
