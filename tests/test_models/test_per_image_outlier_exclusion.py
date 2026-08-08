from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F  # noqa: N812
from traiNNer.models.sr_model import SRModel


def _exclusion_model(*, accum_iters: int = 1) -> SRModel:
    model = object.__new__(SRModel)
    model.per_image_outlier_guard_opt = SimpleNamespace(action="exclude")
    model.accum_iters = accum_iters
    return model


def test_exclusion_contract_accepts_accumulation_and_distillation() -> None:
    model = _exclusion_model(accum_iters=4)
    model.distillation = True

    model._validate_per_image_outlier_exclusion_contract()  # noqa: SLF001


def test_guarded_batch_selection_removes_sample_from_arbitrary_losses() -> None:
    model = _exclusion_model()
    prediction = torch.tensor(
        [[[[1.0]]], [[[100.0]]], [[[3.0]]]], requires_grad=True
    )
    target = torch.zeros_like(prediction)
    keep_mask = torch.tensor([True, False, True])

    selected_prediction = model._select_guarded_batch_tensor(  # noqa: SLF001
        prediction, keep_mask, name="prediction"
    )
    selected_target = model._select_guarded_batch_tensor(  # noqa: SLF001
        target, keep_mask, name="target"
    )
    loss = F.l1_loss(selected_prediction, selected_target) + F.mse_loss(
        selected_prediction, selected_target
    )
    loss.backward()

    assert torch.count_nonzero(prediction.grad[1]).item() == 0
    assert torch.count_nonzero(prediction.grad[[0, 2]]).item() > 0


def test_guarded_accumulation_matches_mean_over_all_survivors() -> None:
    model = _exclusion_model(accum_iters=2)
    reference_samples = 3
    parameter = torch.tensor(0.25, requires_grad=True)
    batches = (
        (
            torch.tensor([1.0, 20.0, 3.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([True, False, True]),
        ),
        (
            torch.tensor([2.0, 4.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([False, True]),
        ),
    )

    total_kept = 0
    for inputs, targets, keep_mask in batches:
        kept = int(keep_mask.sum().item())
        total_kept += kept
        loss = F.mse_loss(
            parameter * inputs[keep_mask], targets[keep_mask]
        )
        weight = model._guarded_accumulation_loss_weight(  # noqa: SLF001
            kept, model.accum_iters, reference_samples
        )
        (loss * weight).backward()

    gradient_scale = model._guarded_accumulation_gradient_scale(  # noqa: SLF001
        total_kept, model.accum_iters, reference_samples
    )
    accumulated_gradient = parameter.grad.detach() * gradient_scale

    expected_parameter = torch.tensor(0.25, requires_grad=True)
    survivor_inputs = torch.cat(
        [inputs[keep_mask] for inputs, _targets, keep_mask in batches]
    )
    survivor_targets = torch.cat(
        [targets[keep_mask] for _inputs, targets, keep_mask in batches]
    )
    F.mse_loss(
        expected_parameter * survivor_inputs, survivor_targets
    ).backward()

    assert accumulated_gradient.item() == pytest.approx(
        expected_parameter.grad.item()
    )



def test_guarded_batch_selection_rejects_non_batch_auxiliary_tensor() -> None:
    model = _exclusion_model()

    with pytest.raises(ValueError, match="batch dimension"):
        model._select_guarded_batch_tensor(  # noqa: SLF001
            torch.tensor(1.0),
            torch.tensor([True, False]),
            name="already-reduced auxiliary loss",
        )
