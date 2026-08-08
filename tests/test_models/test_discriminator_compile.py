# ruff: noqa: SLF001

from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import SGD
from traiNNer.models.sr_model import SRModel


def _make_model(
    *,
    compile_generator: bool,
    compile_discriminator: bool,
    compile_mode: str,
    accum_iters: int,
) -> tuple[SRModel, nn.Parameter, nn.Parameter]:
    model = object.__new__(SRModel)
    model.opt = SimpleNamespace(compile_mode=compile_mode)
    model.use_compile = compile_generator
    model.compile_discriminator = compile_discriminator
    model.accum_iters = accum_iters
    model.cudagraph_stable_gradients = False
    model.cudagraph_stable_discriminator_gradients = False

    generator_parameter = nn.Parameter(torch.ones(2))
    discriminator_parameter = nn.Parameter(torch.ones(3))
    model.optimizer_g = SGD([generator_parameter], lr=0.1)
    model.optimizer_d = SGD([discriminator_parameter], lr=0.1)
    return model, generator_parameter, discriminator_parameter


def test_compiled_discriminator_preallocates_only_its_accumulation_grads() -> None:
    model, generator_parameter, discriminator_parameter = _make_model(
        compile_generator=False,
        compile_discriminator=True,
        compile_mode="max-autotune",
        accum_iters=2,
    )

    model._prepare_cudagraph_gradient_accumulation()

    assert model.cudagraph_stable_gradients is False
    assert model.cudagraph_stable_discriminator_gradients is True
    assert generator_parameter.grad is None
    assert discriminator_parameter.grad is not None
    assert torch.count_nonzero(discriminator_parameter.grad) == 0

    discriminator_parameter.grad.fill_(1)
    assert model.optimizer_d is not None
    model._zero_discriminator_optimizer(model.optimizer_d)

    assert discriminator_parameter.grad is not None
    assert torch.count_nonzero(discriminator_parameter.grad) == 0


def test_non_cudagraph_discriminator_compile_releases_grads() -> None:
    model, _, discriminator_parameter = _make_model(
        compile_generator=False,
        compile_discriminator=True,
        compile_mode="max-autotune-no-cudagraphs",
        accum_iters=2,
    )
    discriminator_parameter.grad = torch.ones_like(discriminator_parameter)

    model._prepare_cudagraph_gradient_accumulation()

    assert model.cudagraph_stable_discriminator_gradients is False
    assert model.optimizer_d is not None
    model._zero_discriminator_optimizer(model.optimizer_d)
    assert discriminator_parameter.grad is None
