"""Small CPU regressions; no user data, CUDA, or compiler kernels."""

from collections.abc import Callable
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from traiNNer.models.sr_model import SRModel
from traiNNer.training_recovery import (
    MAX_CONSECUTIVE_NONFINITE_ATTEMPTS,
    PersistentNonfiniteError,
)
from traiNNer.utils.redux_options import (
    DatasetOptions,
    PathOptions,
    ReduxOptions,
    TrainOptions,
)


class CacheNetwork(nn.Conv2d):
    def __init__(self) -> None:
        super().__init__(3, 3, 1)
        self.prepared = None

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> "CacheNetwork":
        self.prepared = None
        return super()._apply(fn, recurse=recurse)

    def prepare_for_compile(
        self, *, input_shape: tuple[int, ...] | None, input_dtype: torch.dtype
    ) -> None:
        self.prepared = (input_shape, input_dtype)


def make_trainer(*, compiled: bool = False, accum: int = 1) -> SRModel:
    opt = ReduxOptions(
        name="cpu-merge-regression",
        scale=1,
        num_gpu=0,
        path=PathOptions(),
        is_train=True,
        use_compile=compiled,
        compile_mode="default",
        network_g={"type": "cpu-test"},
        datasets={
            "train": DatasetOptions(
                name="unused",
                type="pairedimagedataset",
                batch_size_per_gpu=1,
                lq_size=2,
                accum_iter=accum,
            )
        },
        train=TrainOptions(
            total_iter=10,
            ema_decay=0.99,
            grad_clip=True,
            optim_g={"type": "AdamW", "lr": 1e-3},
            scheduler={"type": "MultiStepLR", "milestones": [2], "gamma": 0.5},
            losses=[{"type": "charbonnierloss", "loss_weight": 1.0}],
        ),
    )
    with (
        patch(
            "traiNNer.models.sr_model.build_network",
            side_effect=lambda _: CacheNetwork(),
        ),
        patch("traiNNer.models.base_model.require_triton"),
        patch(
            "traiNNer.models.base_model.configure_compile_cache_runtime",
            return_value="unused",
        ),
        patch.object(torch, "compile", side_effect=lambda net, **_: net),
    ):
        model = SRModel(opt)
    model.lq = torch.ones(1, 3, 2, 2)
    model.gt = torch.zeros_like(model.lq)
    return model


def test_cache_is_reprimed_after_ema_device_move() -> None:
    model = make_trainer(compiled=True)
    assert model.net_g.prepared == ((1, 3, 2, 2), torch.float32)
    assert model.net_g_ema.initted.device.type == "cpu"
    assert model.net_g_ema.step.device.type == "cpu"
    assert model.grad_clip_max_norm == 1.0


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("populated", [False, True])
def test_rejected_gradient_backs_off_without_training_update(
    bad: float, enabled: bool, populated: bool
) -> None:
    model = make_trainer()
    model.scaler_g = torch.amp.GradScaler("cpu", enabled=enabled, init_scale=16.0)
    if populated:
        model.optimize_parameters(1, 0, True)
        model.update_learning_rate(1)
    weights = deepcopy(model.net_g.state_dict())
    optimizer = deepcopy(model.optimizer_g.state_dict())
    ema = deepcopy(model.net_g_ema.state_dict())
    scheduler = deepcopy(model.schedulers[0].state_dict())
    hook = model.net_g.weight.register_hook(lambda grad: torch.full_like(grad, bad))
    try:
        model.optimize_parameters(1, 0, True)
    finally:
        hook.remove()
    assert not model.training_step_accepted
    assert model.reset_accumulation_requested
    assert model.training_step_rejection_reason == "nonfinite_generator_gradient"
    assert model.scaler_g.get_scale() == (8.0 if enabled else 1.0)
    torch.testing.assert_close(model.optimizer_g.state_dict(), optimizer, rtol=0, atol=0)
    model.update_learning_rate(1)  # The skip flag is an additional scheduler guard.
    assert model.schedulers[0].state_dict() == scheduler
    for name, tensor in weights.items():
        torch.testing.assert_close(
            model.net_g.state_dict()[name], tensor, rtol=0, atol=0
        )
    for name, tensor in ema.items():
        torch.testing.assert_close(
            model.net_g_ema.state_dict()[name], tensor, rtol=0, atol=0
        )
    assert all(p.grad is None for p in model.net_g.parameters())

    # GradScaler's UNSCALED phase must have been cleared for this attempt.
    model.optimize_parameters(1, 0, True)
    assert model.training_step_accepted
    assert not any(model.optimizers_skipped)
    assert model.nan_count == 0
    assert model.optimizer_g.state
    assert any(
        not torch.equal(model.net_g.state_dict()[k], v) for k, v in weights.items()
    )


def test_repeated_overflows_reach_recovery_even_with_finite_loss() -> None:
    model = make_trainer(accum=2)
    model.scaler_g = torch.amp.GradScaler("cpu", init_scale=16.0)
    hook = model.net_g.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
    try:
        for attempt in range(MAX_CONSECUTIVE_NONFINITE_ATTEMPTS):
            model.optimize_parameters(0, 0, False)
            if attempt + 1 == MAX_CONSECUTIVE_NONFINITE_ATTEMPTS:
                with pytest.raises(PersistentNonfiniteError) as caught:
                    model.optimize_parameters(1, 1, True)
            else:
                model.optimize_parameters(1, 1, True)
    finally:
        hook.remove()
    assert caught.value.reason == "nonfinite_generator_gradient"
    assert caught.value.candidate_iter == 1
    assert caught.value.count == MAX_CONSECUTIVE_NONFINITE_ATTEMPTS
    assert not model.optimizer_g.state


def test_nonfinite_loss_discards_partial_accumulation_and_has_typed_limit() -> None:
    model = make_trainer(accum=2)
    model.scaler_g = torch.amp.GradScaler("cpu", init_scale=16.0)
    model.optimize_parameters(0, 0, False)
    assert any(p.grad is not None for p in model.net_g.parameters())
    model.gt.fill_(float("nan"))
    for _ in range(MAX_CONSECUTIVE_NONFINITE_ATTEMPTS - 1):
        model.optimize_parameters(1, 1, True)
    with pytest.raises(PersistentNonfiniteError) as caught:
        model.optimize_parameters(1, 1, True)
    assert caught.value.reason == "nonfinite_generator_loss"
    assert all(p.grad is None for p in model.net_g.parameters())
    assert model.scaler_g.get_scale() == 16.0  # No backward/unscale overflow recorded.
    assert not model.optimizer_g.state
