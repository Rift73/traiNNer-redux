# ruff: noqa: SLF001

import inspect
from unittest.mock import patch

from torch import nn
from traiNNer.archs import drft_arch, drt_arch
from traiNNer.utils.registry import ARCH_REGISTRY


def test_drt_is_native_iln_only() -> None:
    assert not hasattr(drt_arch, "ResidualLayerNorm")
    assert "use_iln" not in inspect.signature(drt_arch.DRT).parameters

    model = drt_arch.DRT(
        img_size=8,
        embed_dim=32,
        depths=(2,),
        num_heads=(1,),
        unshifted_num_heads=(1,),
        unshifted_attention_dim=(32,),
        window_size=4,
        overlap_window_size=6,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        upscale=2,
        num_feat=16,
        rank=32,
        attn_type="masked",
    )

    assert any(
        isinstance(module, drft_arch.ImageLayerNorm)
        for module in model.modules()
    )
    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())
    assert int(model._drt_metadata[0]) == drt_arch.DRT_CHECKPOINT_VERSION


def test_drt_family_uses_the_depth_oriented_geometry() -> None:
    expected = {
        "drt_light": (96, 2, 3),
        "drt_xs": (128, 4, 4),
        "drt_s": (160, 6, 5),
        "drt_m": (192, 8, 6),
        "drt_l": (224, 10, 7),
        "drt_xl": (256, 14, 8),
    }

    sentinel = object()
    for name, contract in expected.items():
        with patch.object(drt_arch, "_build_drt", return_value=sentinel) as build:
            assert getattr(drt_arch, name)() is sentinel

        kwargs = build.call_args.kwargs
        assert (
            kwargs["embed_dim"],
            kwargs["groups"],
            kwargs["full_heads"],
        ) == contract
        assert kwargs["depth"] == 6
        assert kwargs["iln_eps"] == 1e-4
        assert "use_iln" not in kwargs


def test_drt_factories_are_registered() -> None:
    names = ("drt_light", "drt_xs", "drt_s", "drt_m", "drt_l", "drt_xl")
    assert all(ARCH_REGISTRY.get(name) is getattr(drt_arch, name) for name in names)
