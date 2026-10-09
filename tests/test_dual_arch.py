"""Focused CPU contracts for the standalone DUAL3X consolidation."""

import copy
import gc
import io
import os
from collections.abc import Iterator

import pytest
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from traiNNer.archs import build_network
from traiNNer.archs import dual_arch as dual
from traiNNer.utils.registry import ARCH_REGISTRY

PRESETS = [
    ("light", 128, 4),
    ("xs", 128, 4),
    ("s", 160, 6),
    ("m", 192, 6),
    ("l", 224, 10),
    ("xl", 256, 12),
]


@pytest.fixture(scope="module", autouse=True)
def cpu_only() -> Iterator[None]:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(autouse=True)
def release() -> Iterator[None]:
    yield
    gc.collect()


def perturb(model: nn.Module) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.001)


@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("name,width,groups", PRESETS)
def test_factory_geometry_meta(name: str, width: int, groups: int, scale: int) -> None:
    assert ARCH_REGISTRY.get("dual_" + name) is getattr(dual, "dual_" + name)
    with torch.device("meta"):
        model = build_network({"type": "dual_" + name, "scale": scale})
    assert model.embed_dim == width and len(model.groups) == groups
    assert model.num_heads == (width // 32,) * groups
    assert model.depths == tuple(len(g.layers) for g in model.groups)
    assert model.unshuffle == (name == "light")
    assert model.internal_scale == scale * (2 if name == "light" else 1)
    assert (model.xr.hr is not None) == (model.internal_scale >= 2)
    assert (model.xg is not None) == (model.internal_scale >= 4)
    assert len(model.dynamic_conv_names) == groups
    for group in model.groups:
        assert isinstance(group.conv, dual.PremixedDynamicGroupConv3x3)
        for layer in group.layers:
            assert isinstance(layer.ffn.expand, nn.Conv2d)
            assert isinstance(layer.ffn.project, nn.Conv2d)
            assert isinstance(layer.attention.project, nn.Linear)
    if name == "light":
        assert model.selected_original_indices == (
            (0, 1, 2, 3),
            (3, 4, 5),
            (0, 1, 2),
            (0, 1, 2, 3, 4, 5),
        )
        assert model.depths == (4, 3, 3, 6)
    elif name not in ("xs",):
        assert model.depths == (6,) * groups
    with pytest.raises(ValueError, match="scale 1/2/4"):
        getattr(dual, "dual_" + name)(scale=3)


@pytest.mark.parametrize("pu2", [False, True])
@pytest.mark.parametrize("scale", [1, 2, 4])
def test_scales_odd_shapes_gradients_copy_export_and_routing(
    pu2: bool, scale: int
) -> None:
    torch.manual_seed(1024)
    model = dual.DUAL(
        scale=scale,
        embed_dim=32,
        depths=(3, 3),
        layer_indices=((0, 1, 2), (3, 4, 5)),
        pixel_unshuffle=pu2,
        dual_core_checkpoint=False,
    ).double()
    perturb(model)
    with pytest.raises(RuntimeError, match="eval"):
        model.prepare_for_export()
    x = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 3, 5 * scale, 7 * scale)
    out.square().mean().backward()
    assert x.grad.isfinite().all()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )
    group = model.groups[0].conv
    assert group.experts.grad.abs().max() > 0
    assert group.router.fc2.weight.grad.abs().max() > 0
    model.zero_grad(set_to_none=True)
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone = copy.deepcopy(model)
    clone.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    clone.prepare_for_compile(input_dtype=torch.bfloat16)
    clone.prepare_for_export()
    assert all(
        m.g_attention_impl == "sdpa"
        for m in clone.modules()
        if isinstance(m, dual._SpatialAttention)  # noqa: SLF001 - verify export hook
    )
    assert not any(isinstance(m, dual.EDBBReparamConv3x3) for m in clone.modules())
    identities = [id(m) for m in clone.modules()]
    clone.prepare_for_export()
    assert identities == [id(m) for m in clone.modules()]
    with torch.no_grad():
        for h, w in ((1, 1), (5, 7), (9, 11)):
            q = torch.randn(2, 3, h, w, dtype=torch.float64)
            a, b = model(q), clone(q)
            assert a.shape == (2, 3, h * scale, w * scale)
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
        conv = clone.groups[0].conv
        q = torch.randn(2, 32, 3, 5, dtype=torch.float64)
        assert not torch.equal(conv.coefficients(q)[0], conv.coefficients(q)[1])
        assert (conv(q) - conv.base(q)).abs().max() > 1e-8


@pytest.mark.parametrize("batch", [1, 3])
def test_premixed_explicit_expert_oracle_all_gradients(batch: int) -> None:
    torch.manual_seed(1024)
    mod = dual.PremixedDynamicGroupConv3x3(
        dual.EDBBReparamConv3x3(nn.Conv2d(6, 6, 3, padding=1).double())
    )
    perturb(mod)
    x = torch.randn(batch, 6, 5, 7, dtype=torch.float64, requires_grad=True)
    out = mod(x)
    ref = mod.base(x)
    co = mod.coefficients(x)
    for i, kernel in enumerate(mod.experts.reshape(4, 6, 6, 3, 3)):
        ref = ref + co[:, i, None] * F.conv2d(x, kernel, padding=1)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-10)
    probe = torch.randn_like(out)
    variables = (x, *mod.parameters())
    a = torch.autograd.grad((out * probe).sum(), variables, retain_graph=True)
    b = torch.autograd.grad((ref * probe).sum(), variables)
    for aa, bb in zip(a, b, strict=True):
        torch.testing.assert_close(aa, bb, atol=1e-11, rtol=1e-9)


def test_light_real_width_bf16_channelslast() -> None:
    torch.manual_seed(1024)
    model = dual.dual_light(dual_core_checkpoint=False).to(
        memory_format=torch.channels_last
    )
    perturb(model)
    x = (
        torch.randn(2, 3, 5, 7)
        .to(memory_format=torch.channels_last)
        .requires_grad_(True)
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x)
    assert out.shape == (2, 3, 20, 28) and out.isfinite().all()
    out.float().square().mean().backward()
    assert x.grad.isfinite().all()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )


@pytest.mark.parametrize("name,width,groups", PRESETS)
def test_real_factory_forward_fold(name: str, width: int, groups: int) -> None:
    torch.manual_seed(1024)
    model = getattr(dual, "dual_" + name)(dual_core_checkpoint=False).eval()
    with torch.no_grad():
        for n in model.dynamic_conv_names:
            model.get_submodule(n).experts.normal_(std=0.001)
        x = torch.randn(1, 3, 4, 4)
        expected = model(x)
        assert expected.shape == (1, 3, 16, 16)
        before = sum(p.numel() for p in model.parameters())
        model.prepare_for_export()
        actual = model(x)
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
        assert actual.isfinite().all()
        print(
            "DUAL_FACTORY",
            name,
            width,
            groups,
            before,
            sum(p.numel() for p in model.parameters()),
            (actual - expected).abs().max().item(),
            flush=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dynamic_experts": True},
        {"dynamic_experts": 1},
        {"depths": (3, 3)},
        {"depths": (2, 2), "layer_indices": ((1, 2), (0, 1))},
        {"depths": (2, 2), "layer_indices": ((0, 1), (5, 5))},
        {"depths": (2, 2), "layer_indices": ((0, 1),)},
    ],
)
def test_invalid_build(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        dual.DUAL(**kwargs)
