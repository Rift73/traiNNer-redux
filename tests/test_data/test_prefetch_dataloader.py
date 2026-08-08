import torch
from traiNNer.data.prefetch_dataloader import _memory_format_for_tensor


def test_memory_format_for_tensor_keeps_channels_last_for_rank4() -> None:
    tensor = torch.zeros(2, 3, 8, 8)

    assert _memory_format_for_tensor(tensor, torch.channels_last) is torch.channels_last


def test_memory_format_for_tensor_falls_back_for_non_rank4() -> None:
    tensor = torch.zeros(2, 21, 21)

    assert _memory_format_for_tensor(tensor, torch.channels_last) is torch.preserve_format
