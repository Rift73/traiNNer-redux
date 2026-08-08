import pytest
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grads_with_norm_
from traiNNer.models.sr_model import _get_total_gradient_norm


def make_parameters() -> list[nn.Parameter]:
    first = nn.Parameter(torch.zeros(2, 2))
    second = nn.Parameter(torch.zeros(3))
    unused = nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([[3.0, 4.0], [0.0, 12.0]])
    second.grad = torch.tensor([5.0, 0.0, 0.0])
    return [first, second, unused]


def test_fused_total_gradient_norm_matches_legacy_reference() -> None:
    parameters = make_parameters()
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    reference = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(gradient, 2) for gradient in gradients])
    ).detach()

    actual = _get_total_gradient_norm(parameters)

    torch.testing.assert_close(actual, reference)
    assert not actual.requires_grad


def test_precomputed_norm_clipping_matches_clip_grad_norm() -> None:
    parameters = make_parameters()
    reference_parameters = make_parameters()

    total_norm = _get_total_gradient_norm(parameters)
    clip_grads_with_norm_(
        parameters,
        max_norm=2.0,
        total_norm=total_norm,
        foreach=None,
    )
    reference_norm = clip_grad_norm_(
        reference_parameters,
        max_norm=2.0,
        error_if_nonfinite=False,
        foreach=None,
    )

    torch.testing.assert_close(total_norm, reference_norm)
    for parameter, reference_parameter in zip(
        parameters, reference_parameters, strict=True
    ):
        if parameter.grad is None:
            assert reference_parameter.grad is None
        else:
            assert reference_parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_fused_total_gradient_norm_propagates_nonfinite(bad_value: float) -> None:
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([1.0, bad_value])

    total_norm = _get_total_gradient_norm([parameter])

    assert not torch.isfinite(total_norm).item()


def test_fused_total_gradient_norm_rejects_missing_gradients() -> None:
    parameter = nn.Parameter(torch.zeros(1))

    with pytest.raises(RuntimeError, match="without gradients"):
        _get_total_gradient_norm([parameter])
