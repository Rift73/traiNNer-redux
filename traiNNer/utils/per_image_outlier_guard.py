from dataclasses import dataclass

import torch
from torch import Tensor

PerImageOutlierTensorStats = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class PerImageOutlierStats:
    """Detached per-image reconstruction-error statistics for shadow guarding."""

    charbonnier_mean: Tensor
    max_abs_error: Tensor
    batch_median: Tensor
    mean_threshold: Tensor
    nonfinite: Tensor
    outliers: Tensor


def make_per_image_keep_weight(outliers: Tensor, reference: Tensor) -> Tensor:
    """Create a one-channel spatial weight that removes classified samples.

    The expanded view matches traiNNer's weighted mean-loss contract: its
    denominator counts every kept spatial location and channel, so removing a
    sample does not dilute the remaining samples' gradient.
    """
    if outliers.ndim != 1 or outliers.shape[0] != reference.shape[0]:
        raise ValueError("outlier mask must contain one value per reference sample")
    if reference.ndim < 2:
        raise ValueError("reference must have batch and channel dimensions")

    spatial_shape = reference.shape[2:]
    base = (~outliers.detach()).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    base = base.reshape(reference.shape[0], 1, *(1 for _ in spatial_shape))
    return base.expand(reference.shape[0], 1, *spatial_shape)


def evaluate_per_image_outlier_tensors(
    prediction: Tensor,
    target: Tensor,
    *,
    charbonnier_epsilon: float,
    mean_absolute_floor: float,
    median_multiple: float,
    max_absolute_error: float,
) -> PerImageOutlierTensorStats:
    """Return branch-free FP32 tensors for per-image outlier classification.

    Keeping this function tensor-only lets TorchInductor capture it as one
    dedicated full graph. Transactional rejection, exclusion, and logging stay
    in the trainer's Python control flow.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "per-image outlier guard requires prediction and target shapes to match"
        )
    if prediction.ndim < 2 or prediction.shape[0] == 0:
        raise ValueError("per-image outlier guard requires a non-empty batch")

    with (
        torch.no_grad(),
        torch.autocast(
            device_type=prediction.device.type,
            enabled=False,
        ),
    ):
        difference = prediction.detach().float() - target.detach().float()
        flattened_difference = difference.flatten(1)
        max_abs_error = flattened_difference.abs().amax(1)

        # Reuse the detached FP32 difference buffer for the Charbonnier
        # reduction. This avoids retaining separate full-resolution absolute,
        # squared, and square-root tensors on every training microbatch.
        difference.square_().add_(charbonnier_epsilon).sqrt_()
        charbonnier_mean = difference.flatten(1).mean(1)
        batch_median = charbonnier_mean.median()
        mean_threshold = torch.clamp_min(
            batch_median * median_multiple,
            mean_absolute_floor,
        )
        nonfinite = ~(torch.isfinite(charbonnier_mean) & torch.isfinite(max_abs_error))
        outliers = (
            nonfinite
            | (max_abs_error > max_absolute_error)
            | (charbonnier_mean > mean_threshold)
        )

    return (
        charbonnier_mean,
        max_abs_error,
        batch_median,
        mean_threshold,
        nonfinite,
        outliers,
    )


def evaluate_per_image_outliers(
    prediction: Tensor,
    target: Tensor,
    *,
    charbonnier_epsilon: float,
    mean_absolute_floor: float,
    median_multiple: float,
    max_absolute_error: float,
) -> PerImageOutlierStats:
    """Classify destructive per-image reconstruction outliers in FP32."""
    return PerImageOutlierStats(
        *evaluate_per_image_outlier_tensors(
            prediction,
            target,
            charbonnier_epsilon=charbonnier_epsilon,
            mean_absolute_floor=mean_absolute_floor,
            median_multiple=median_multiple,
            max_absolute_error=max_absolute_error,
        )
    )
