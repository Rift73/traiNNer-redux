"""CPU output/control-flow contracts for the isolated-harness optimizations."""
# ruff: noqa: SLF001

from contextlib import nullcontext
from unittest.mock import PropertyMock, patch

import pytest
import torch
from traiNNer.data import degradations
from traiNNer.data.prefetch_dataloader import CUDAPrefetcher
from traiNNer.models.realesrgan_model import RealESRGANModel, _collated_any


@pytest.mark.parametrize("kind", ["gaussian", "poisson"])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("rounds", [False, True])
@pytest.mark.parametrize("gray", [0.0, 1.0])
def test_zero_noise_preserves_output_processing_without_rng(
    kind: str, clip: bool, rounds: bool, gray: float
) -> None:
    image = torch.linspace(-0.2, 1.2, 24).reshape(2, 3, 2, 2)
    before = torch.get_rng_state().clone()
    func = getattr(degradations, f"random_add_{kind}_noise_pt")
    with patch.object(
        degradations, f"random_generate_{kind}_noise_pt", side_effect=AssertionError
    ):
        actual = func(image, (0, 0), gray_prob=gray, clip=clip, rounds=rounds)
    expected = image
    if clip and rounds:
        expected = (image * 255.0).round().clamp(0, 255) / 255.0
    elif clip:
        expected = image.clamp(0, 1)
    elif rounds:
        expected = (image * 255.0).round() / 255.0
    assert torch.equal(actual, expected)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["gaussian", "poisson"])
def test_nonzero_range_keeps_noise_generator(kind: str) -> None:
    image = torch.zeros(1, 3, 2, 2)
    with patch.object(
        degradations,
        f"random_generate_{kind}_noise_pt",
        return_value=torch.ones_like(image),
    ) as noise:
        result = getattr(degradations, f"random_add_{kind}_noise_pt")(
            image, (0, 1), clip=False
        )
    noise.assert_called_once()
    assert torch.equal(result, torch.ones_like(image))


def test_prefetch_keeps_crop_metadata_on_host() -> None:
    prefetch = object.__new__(CUDAPrefetcher)
    flags = torch.tensor([False, True])
    image = torch.zeros(1, 3, 2, 2)
    prefetch.loader = iter(
        [{"gt": image, "otf_padded_height": flags, "otf_padded_width": flags}]
    )
    prefetch.stream = None
    prefetch.device = torch.device("cpu")
    prefetch.memory_format = torch.preserve_format
    original = torch.Tensor.to
    transferred = []

    def transfer(tensor: torch.Tensor, *args: object, **kwargs) -> torch.Tensor:
        transferred.append(tensor)
        return original(tensor, *args, **kwargs)

    with (
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch.object(torch.Tensor, "to", transfer),
    ):
        prefetch.preload()
    assert len(transferred) == 1 and transferred[0] is image
    assert prefetch.batch["otf_padded_height"] is flags
    assert _collated_any(flags)
    assert not _collated_any(None)


@pytest.mark.parametrize("cuda_path", [False, True])
def test_queue_uses_one_shared_permutation_and_retains_transfer_source(
    cuda_path: bool,
) -> None:
    model = object.__new__(RealESRGANModel)
    model.queue_size = model.queue_ptr = 4
    model.queue_lr = torch.arange(4.0).reshape(4, 1, 1, 1)
    model.queue_gt = model.queue_lr + 10
    model.lq = torch.tensor([20.0, 21.0]).reshape(2, 1, 1, 1)
    model.gt = model.lq + 10
    order = torch.tensor([3, 1, 0, 2])
    original = torch.Tensor.to
    transfers = []

    def transfer(tensor: torch.Tensor, *args: object, **kwargs) -> torch.Tensor:
        transfers.append((tensor, kwargs))
        return original(tensor, *args, **kwargs)

    # Pinning and the async transfer are mocked; this does not validate CUDA timing.
    with (
        patch("torch.randperm", return_value=order),
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=cuda_path
        ),
        patch.object(torch.Tensor, "pin_memory", lambda t: t),
        patch.object(torch.Tensor, "to", transfer),
    ):
        model._dequeue_and_enqueue()
    if cuda_path:
        assert model._queue_index_host is order
        assert len(transfers) == 1 and transfers[0][0] is order
        assert transfers[0][1]["non_blocking"] is True
    else:
        assert getattr(model, "_queue_index_host", None) is None
        assert not transfers
    assert torch.equal(model.lq.flatten(), torch.tensor([3.0, 1.0]))
    assert torch.equal(model.gt.flatten(), torch.tensor([13.0, 11.0]))
    assert torch.equal(model.queue_lr.flatten(), torch.tensor([20.0, 21.0, 0.0, 2.0]))
