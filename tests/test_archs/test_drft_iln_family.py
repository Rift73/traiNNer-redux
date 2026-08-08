from unittest.mock import patch

import torch
from torch import nn
from traiNNer.archs import drft_arch


def test_drft_uses_iln_at_every_normalization_site() -> None:
    model = drft_arch.DRFT(
        img_size=8,
        embed_dim=32,
        depths=(2,),
        num_heads=(1,),
        unshifted_num_heads=(1,),
        unshifted_attention_dim=(16,),
        window_size=4,
        overlap_window_size=6,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        upscale=2,
        num_feat=16,
        rank=32,
        attn_type="masked",
    )

    assert sum(
        isinstance(module, drft_arch.ImageLayerNorm)
        for module in model.modules()
    ) == 6
    assert isinstance(model.norm, drft_arch.AffineTransform)
    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())


def test_drft_family_has_balanced_factory_contract() -> None:
    expected = {
        "drft_nano": (32, 1, 2, 1, 1, 16),
        "drft_micro": (64, 1, 6, 2, 1, 32),
        "drft_light": (96, 2, 6, 3, 1, 48),
        "drft_xs": (128, 4, 6, 4, 2, 64),
        "drft_s": (160, 6, 6, 5, 3, 96),
        "drft_m": (192, 8, 6, 6, 3, 96),
        "drft_l": (224, 10, 6, 7, 4, 128),
        "drft_xl": (256, 14, 6, 8, 4, 128),
    }

    sentinel = object()
    for name, contract in expected.items():
        with patch.object(drft_arch, "_build_drft", return_value=sentinel) as build:
            assert getattr(drft_arch, name)() is sentinel

        kwargs = build.call_args.kwargs
        observed = (
            kwargs["embed_dim"],
            kwargs["groups"],
            kwargs["depth"],
            kwargs["full_heads"],
            kwargs["unshifted_heads"],
            kwargs["unshifted_attention_dim"],
        )
        assert observed == contract
        assert kwargs["rank"] == 32
        assert kwargs["iln_eps"] == 1e-4
        assert "use_iln" not in kwargs
        if name == "drft_xl":
            assert kwargs["rhag_layer_scale_init"] == 1e-4
        else:
            assert "rhag_layer_scale_init" not in kwargs


def test_rhag_outer_layerscale_is_opt_in() -> None:
    common = {
        "img_size": 8,
        "embed_dim": 32,
        "depths": (2,),
        "num_heads": (1,),
        "unshifted_num_heads": (1,),
        "unshifted_attention_dim": (16,),
        "window_size": 4,
        "overlap_window_size": 6,
        "mlp_ratio": 2.0,
        "drop_path_rate": 0.0,
        "upscale": 2,
        "num_feat": 16,
        "rank": 32,
        "attn_type": "masked",
    }

    unchanged = drft_arch.DRFT(**common)
    assert isinstance(unchanged.layers[0].ls_rhag, nn.Identity)
    assert not any("ls_rhag" in key for key in unchanged.state_dict())
    assert unchanged.compile_cache_key_tag.startswith(
        "drft-iln-family-v11-raw-aot-ocab-flex-op-"
    )
    assert unchanged.compile_cache_key_tag == drft_arch.DRFT(
        **common
    ).compile_cache_key_tag

    stabilized = drft_arch.DRFT(**common, rhag_layer_scale_init=1e-4)
    assert isinstance(stabilized.layers[0].ls_rhag, drft_arch.LayerScale)
    assert stabilized.compile_cache_key_tag.startswith(
        "drft-iln-family-v11-raw-aot-ocab-flex-op-"
    )
    assert stabilized.compile_cache_key_tag != unchanged.compile_cache_key_tag
    assert torch.equal(
        stabilized.layers[0].ls_rhag.gamma,
        torch.full((32,), 1e-4),
    )


def test_rhag_outer_layerscale_gates_the_complete_group_branch() -> None:
    class DoubleResidual(nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            _x_size: tuple[int, int],
            _hybrid_ctx: object | None = None,
        ) -> torch.Tensor:
            return x * 2

    group = drft_arch.RHAG(
        dim=8,
        input_resolution=(2, 2),
        depth=1,
        num_heads=1,
        unshifted_num_heads=1,
        unshifted_attention_dim=8,
        window_size=2,
        overlap_window_size=4,
        drop_path=0.0,
        dense_skip=False,
        rank=32,
        rhag_layer_scale_init=0.25,
    )
    group.residual_group = DoubleResidual()
    group.conv = nn.Identity()

    x = torch.ones(1, 4, 8)
    assert torch.equal(group(x, (2, 2)), x * 1.5)


def test_drft_exposes_no_optional_iln_compatibility_factories() -> None:
    assert not hasattr(drft_arch, "drft_micro_iln")
    assert not hasattr(drft_arch, "drft_micro_iln_deep4")
    assert not hasattr(drft_arch, "drft_light_iln")
