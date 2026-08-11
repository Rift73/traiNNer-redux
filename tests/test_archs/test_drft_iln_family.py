# ruff: noqa: SLF001

from unittest.mock import patch

import torch
from torch import nn
from traiNNer.archs import drft_arch
from traiNNer.utils.registry import ARCH_REGISTRY


def test_drft_uses_iln_at_every_normalization_site() -> None:
    model = drft_arch.DRFT(
        img_size=8,
        embed_dim=32,
        depths=(2,),
        num_heads=(1,),
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


def test_iln_caps_only_the_abnormal_training_tail() -> None:
    layer = drft_arch.ImageLayerNorm(dim=1, eps=0.0)
    layer.train()

    # Mirror the model pre-hook that initializes the training reference before
    # the first forward. A raw standard deviation of 3 is capped at 2.
    layer.update_std_ema()
    x = torch.tensor([[[-3.0], [-3.0], [3.0], [3.0]]])
    normalized, scale = layer(x)

    assert torch.equal(normalized, torch.tensor([[[-1.0], [-1.0], [1.0], [1.0]]]))
    assert torch.equal(scale, torch.tensor([[[2.0]]]))
    assert torch.equal(layer._last_std_buf, torch.tensor([3.0]))

    # The guarded update rejects the abnormal observation instead of letting it
    # raise the cap for the following image.
    layer.update_std_ema()
    assert torch.equal(layer.std_ema, torch.ones(1))

    normal = torch.tensor([[[-1.5], [-1.5], [1.5], [1.5]]])
    _, normal_scale = layer(normal)
    assert torch.equal(normal_scale, torch.tensor([[[1.5]]]))


def test_iln_eval_is_raw_and_does_not_mutate_training_state() -> None:
    layer = drft_arch.ImageLayerNorm(dim=2, eps=1e-4)
    layer.train()
    layer.update_std_ema()
    layer(torch.tensor([[[-1.0, -2.0], [1.0, 2.0]]]))

    ema_before = layer.std_ema.clone()
    observation_before = layer._last_std_buf.clone()
    x = torch.tensor([[[-9.0, -3.0], [3.0, 9.0]]])

    layer.eval()
    output, scale = layer(x)
    mean = x.mean(dim=(1, 2), keepdim=True)
    centered = x - mean
    expected_std = torch.sqrt(
        (centered * centered).mean(dim=(1, 2), keepdim=True) + layer.eps
    )
    expected_output = centered / expected_std * layer.weight + layer.bias

    assert torch.equal(output, expected_output)
    assert torch.equal(scale, expected_std)
    assert torch.equal(layer.std_ema, ema_before)
    assert torch.equal(layer._last_std_buf, observation_before)
    assert set(layer.state_dict()) == {"weight", "bias"}


def test_drft_family_has_balanced_factory_contract() -> None:
    expected = {
        "drft_nano": (32, 1, 2, 1),
        "drft_micro": (64, 1, 6, 2),
        "drft_light": (96, 2, 6, 3),
        "drft_xs": (128, 4, 6, 4),
        "drft_s": (160, 6, 6, 5),
        "drft_m": (192, 8, 6, 6),
        "drft_l": (224, 10, 6, 7),
        "drft_xl": (256, 14, 6, 8),
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
        )
        assert observed == contract
        assert "unshifted_heads" not in kwargs
        assert "unshifted_attention_dim" not in kwargs
        assert kwargs["rank"] == 32
        assert kwargs["iln_eps"] == 1e-4
        assert "use_iln" not in kwargs
        if name == "drft_xl":
            assert kwargs["rhag_layer_scale_init"] == 1e-4
        else:
            assert "rhag_layer_scale_init" not in kwargs


def test_canonical_drft_uses_full_width_attention_and_original_ocab_overlap() -> None:
    model = drft_arch.drft_nano(scale=2, drop_path_rate=0.0)

    assert model.overlap_window_size == 24
    assert model.unshifted_num_heads == model.num_heads
    assert model.unshifted_attention_dim == (model.embed_dim,) * len(model.depths)
    assert not any(
        isinstance(module, drft_arch.NarrowWindowAttention)
        for module in model.modules()
    )


def test_drft_teacher_family_inherits_the_canonical_attention_contract() -> None:
    expected = {
        "drft_nano_teacher": (32, 2, 4, 1),
        "drft_micro_teacher": (64, 2, 6, 2),
        "drft_light_teacher": (96, 4, 6, 3),
        "drft_xs_teacher": (128, 6, 6, 4),
        "drft_s_teacher": (160, 9, 6, 5),
        "drft_m_teacher": (192, 12, 6, 6),
        "drft_l_teacher": (224, 15, 6, 7),
        "drft_xl_teacher": (256, 21, 6, 8),
    }

    sentinel = object()
    for name, contract in expected.items():
        with patch.object(
            drft_arch,
            "_build_drft_teacher",
            return_value=sentinel,
        ) as build:
            assert getattr(drft_arch, name)() is sentinel

        kwargs = build.call_args.kwargs
        observed = (
            kwargs["embed_dim"],
            kwargs["groups"],
            kwargs["depth"],
            kwargs["full_heads"],
        )
        assert observed == contract
        assert "unshifted_heads" not in kwargs
        assert "unshifted_attention_dim" not in kwargs


def test_only_canonical_drft_names_are_registered() -> None:
    names = (
        "drft_nano",
        "drft_micro",
        "drft_light",
        "drft_xs",
        "drft_s",
        "drft_m",
        "drft_l",
        "drft_xl",
    )
    assert all(ARCH_REGISTRY.get(name) is getattr(drft_arch, name) for name in names)
    assert "drft_m_safe" not in ARCH_REGISTRY


def test_rhag_outer_layerscale_is_opt_in() -> None:
    common = {
        "img_size": 8,
        "embed_dim": 32,
        "depths": (2,),
        "num_heads": (1,),
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
        "drft-iln-family-v18-sdpa-ocab-layout-fusion-train-ema-cap-"
    )
    assert unchanged.compile_cache_key_tag == drft_arch.DRFT(
        **common
    ).compile_cache_key_tag

    stabilized = drft_arch.DRFT(**common, rhag_layer_scale_init=1e-4)
    assert isinstance(stabilized.layers[0].ls_rhag, drft_arch.LayerScale)
    assert stabilized.compile_cache_key_tag.startswith(
        "drft-iln-family-v18-sdpa-ocab-layout-fusion-train-ema-cap-"
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
