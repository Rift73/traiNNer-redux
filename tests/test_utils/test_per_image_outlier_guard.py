import pytest
import torch
from traiNNer.losses.basic_loss import CharbonnierLoss
from traiNNer.utils.per_image_outlier_guard import (
    PerImageOutlierStats,
    evaluate_per_image_outlier_tensors,
    evaluate_per_image_outliers,
    make_per_image_keep_weight,
)


def evaluate(prediction: torch.Tensor, target: torch.Tensor) -> PerImageOutlierStats:
    return evaluate_per_image_outliers(
        prediction,
        target,
        charbonnier_epsilon=1e-12,
        mean_absolute_floor=0.15,
        median_multiple=8.0,
        max_absolute_error=4.0,
    )


def test_single_pathological_image_isolated_from_normal_batch() -> None:
    target = torch.zeros(32, 3, 4, 4)
    prediction = target.clone()
    prediction[0] = 1.0
    prediction[0, 0, 0, 0] = 8.0

    stats = evaluate(prediction, target)

    assert stats.outliers.nonzero(as_tuple=False).flatten().tolist() == [0]
    assert stats.nonfinite.sum().item() == 0
    assert stats.max_abs_error[0].item() == 8.0
    assert stats.mean_threshold.item() == pytest.approx(0.15)


def test_uniform_early_training_error_is_not_a_relative_outlier() -> None:
    target = torch.zeros(32, 3, 4, 4)
    prediction = torch.full_like(target, 0.5)

    stats = evaluate(prediction, target)

    assert not stats.outliers.any().item()
    assert stats.batch_median.item() == 0.5
    assert stats.mean_threshold.item() == 4.0


def test_nonfinite_image_is_always_an_outlier() -> None:
    target = torch.zeros(4, 3, 2, 2)
    prediction = target.clone()
    prediction[2, 1, 0, 0] = torch.nan

    stats = evaluate(prediction, target)

    assert stats.nonfinite.tolist() == [False, False, True, False]
    assert stats.outliers.tolist() == [False, False, True, False]


def test_all_nonfinite_batch_preserves_nonfinite_threshold() -> None:
    target = torch.zeros(4, 3, 2, 2)
    prediction = torch.full_like(target, torch.nan)

    stats = evaluate(prediction, target)

    assert torch.isnan(stats.mean_threshold).item()
    assert stats.nonfinite.all().item()
    assert stats.outliers.all().item()


def test_shadow_statistics_are_detached_from_training_graph() -> None:
    target = torch.zeros(2, 3, 2, 2)
    prediction = torch.ones_like(target, requires_grad=True)

    stats = evaluate(prediction, target)

    assert all(
        not value.requires_grad
        for value in (
            stats.charbonnier_mean,
            stats.max_abs_error,
            stats.batch_median,
            stats.mean_threshold,
            stats.nonfinite,
            stats.outliers,
        )
    )


def test_tensor_detector_is_fullgraph_compilable() -> None:
    def detector(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return evaluate_per_image_outlier_tensors(
            prediction,
            target,
            charbonnier_epsilon=1e-12,
            mean_absolute_floor=0.15,
            median_multiple=8.0,
            max_absolute_error=4.0,
        )

    compiled_detector = torch.compile(
        detector,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    target = torch.zeros(4, 3, 4, 4)
    prediction = torch.full_like(target, 0.25, requires_grad=True)
    with torch.no_grad():
        prediction[1, 0, 0, 0] = 8.0

    expected = detector(prediction, target)
    actual = compiled_detector(prediction, target)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
        assert not actual_tensor.requires_grad


def test_keep_weight_removes_only_outlier_gradient_without_dilution() -> None:
    target = torch.zeros(4, 3, 2, 2)
    prediction = torch.full_like(target, 0.1, requires_grad=True)
    with torch.no_grad():
        prediction[0] = 8.0

    stats = evaluate(prediction, target)
    weight = make_per_image_keep_weight(stats.outliers, prediction)
    loss = CharbonnierLoss(loss_weight=1.0)(prediction, target, weight=weight)
    reference_loss = CharbonnierLoss(loss_weight=1.0)(
        prediction[1:], target[1:]
    )

    assert stats.outliers.tolist() == [True, False, False, False]
    assert loss.item() == pytest.approx(reference_loss.item())
    loss.backward()
    assert torch.count_nonzero(prediction.grad[0]).item() == 0
    assert torch.count_nonzero(prediction.grad[1:]).item() > 0
