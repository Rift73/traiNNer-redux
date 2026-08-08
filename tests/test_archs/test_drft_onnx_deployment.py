# ruff: noqa: SLF001

import copy

import pytest
import torch
from convert_to_onnx import resolve_architecture_export_example_shape
from traiNNer.archs.drft_arch import (
    DRFT,
    OCAB,
    EDBBConvBlock,
    NarrowWindowAttention,
    WindowAttentionRFB,
    _DRFTTensorRTMixedPrecision,
    _StaticRankFactoredBias,
)


def _tiny_drft() -> DRFT:
    return DRFT(
        img_size=8,
        embed_dim=32,
        depths=(2,),
        num_heads=(2,),
        unshifted_num_heads=(1,),
        window_size=4,
        overlap_window_size=6,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        upscale=2,
        num_feat=16,
        rank=16,
        attn_type="masked",
    ).eval()


def test_hybrid_attention_contract_reaches_ocab() -> None:
    model = DRFT(
        img_size=8,
        embed_dim=32,
        depths=(2,),
        num_heads=(2,),
        unshifted_num_heads=(1,),
        window_size=4,
        overlap_window_size=6,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        upscale=2,
        num_feat=16,
        rank=16,
        attn_type="hybrid",
    )

    assert model.layers[0].residual_group.ocab.attn_type == "hybrid"


def _edbb_branch_forward(module: EDBBConvBlock, sample: torch.Tensor) -> torch.Tensor:
    return (
        module.conv3x3(sample)
        + module.conv1x1(sample)
        + module.conv1x1_3x3(sample)
        + module.conv1x1_sbx(sample)
        + module.conv1x1_sby(sample)
        + module.conv1x1_lpl(sample)
        + sample
    )


def test_dynamic_drft_export_uses_96px_example_without_fixing_axes() -> None:
    model = _tiny_drft()
    shape = resolve_architecture_export_example_shape(
        model,
        (1, 3, 32, 32),
        (False, False, True, True),
    )
    assert shape == (1, 3, 96, 96)
    assert resolve_architecture_export_example_shape(
        model,
        (1, 3, 48, 64),
        (False, False, False, False),
    ) == (1, 3, 48, 64)


def test_eval_uses_reversible_edbb_fast_path() -> None:
    torch.manual_seed(7)
    module = EDBBConvBlock(8)
    sample = torch.rand(1, 8, 8, 8)
    state_keys = tuple(module.state_dict())

    with torch.inference_mode():
        expected = _edbb_branch_forward(module, sample)

    module.eval()
    assert module._eval_folded_weight is not None
    assert module._eval_folded_bias is not None
    assert not module._is_folded
    assert module._folded_conv is None
    assert hasattr(module, "conv3x3")
    assert tuple(module.state_dict()) == state_keys

    with torch.inference_mode():
        actual = module(sample)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    module.train()
    assert module._eval_folded_weight is None
    assert module._eval_folded_bias is None
    assert tuple(module.state_dict()) == state_keys


def test_repeated_eval_refreshes_edbb_cache_after_parameter_update() -> None:
    torch.manual_seed(7)
    module = EDBBConvBlock(8).eval()
    sample = torch.rand(1, 8, 8, 8)
    old_weight = module._eval_folded_weight.detach().clone()

    with torch.no_grad():
        module.conv3x3.weight.add_(0.25)
        expected = _edbb_branch_forward(module, sample)

    module.eval()
    assert not torch.equal(module._eval_folded_weight, old_weight)
    with torch.inference_mode():
        actual = module(sample)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_edbb_training_reparameterization_preserves_output_and_gradients() -> None:
    torch.manual_seed(7)
    folded = EDBBConvBlock(8).double().train()
    branched = copy.deepcopy(folded)
    folded_input = torch.rand(2, 8, 7, 7, dtype=torch.float64, requires_grad=True)
    branched_input = folded_input.detach().clone().requires_grad_(True)

    folded_output = folded(folded_input)
    branched_output = _edbb_branch_forward(branched, branched_input)
    torch.testing.assert_close(folded_output, branched_output, rtol=1e-12, atol=1e-12)

    output_gradient = torch.randn_like(folded_output)
    folded_output.backward(output_gradient)
    branched_output.backward(output_gradient)
    torch.testing.assert_close(
        folded_input.grad,
        branched_input.grad,
        rtol=1e-11,
        atol=1e-11,
    )
    for (folded_name, folded_parameter), (branched_name, branched_parameter) in zip(
        folded.named_parameters(), branched.named_parameters(), strict=True
    ):
        assert folded_name == branched_name
        if folded_parameter.grad is None or branched_parameter.grad is None:
            present_gradient = (
                branched_parameter.grad
                if folded_parameter.grad is None
                else folded_parameter.grad
            )
            assert present_gradient is not None
            assert present_gradient.abs().max() <= 1e-12
            continue
        torch.testing.assert_close(
            folded_parameter.grad,
            branched_parameter.grad,
            rtol=1e-11,
            atol=1e-11,
        )


def test_eval_cache_refreshes_after_state_dict_load() -> None:
    torch.manual_seed(7)
    module = EDBBConvBlock(8).eval()
    replacement = EDBBConvBlock(8)
    with torch.no_grad():
        replacement.conv3x3.bias.add_(1.0)
    module.load_state_dict(replacement.state_dict())
    sample = torch.rand(1, 8, 8, 8)

    with torch.inference_mode():
        expected = _edbb_branch_forward(module, sample)
        actual = module(sample)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_drft_eval_train_transition_preserves_training_graph() -> None:
    model = _tiny_drft()
    edbb = [
        module for module in model.modules()
        if isinstance(module, EDBBConvBlock)
    ]
    state_keys = tuple(model.state_dict())
    assert edbb and all(module._eval_folded_weight is not None for module in edbb)
    assert not any(module._is_folded for module in edbb)

    model.train()
    assert all(module._eval_folded_weight is None for module in edbb)
    assert all(hasattr(module, "conv3x3") for module in edbb)
    assert tuple(model.state_dict()) == state_keys

    output = model(torch.rand(1, 3, 8, 8))
    output.mean().backward()
    assert all(module.conv3x3.weight.grad is not None for module in edbb)


def test_prepare_for_onnx_export_is_exact_and_non_mutating() -> None:
    torch.manual_seed(7)
    model = _tiny_drft()
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    sample = torch.rand(1, 3, 8, 8)

    with torch.inference_mode():
        expected = model(sample)
    deployment = model.prepare_for_onnx_export(sample.shape)
    with torch.inference_mode():
        actual = deployment(sample)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert model._onnx_deployment_input_shape is None
    assert tuple(model.state_dict()) == tuple(original_state)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original_state[name])

    edbb = [
        module for module in deployment.modules()
        if isinstance(module, EDBBConvBlock)
    ]
    assert edbb and all(module._is_folded for module in edbb)
    assert sum(
        isinstance(module, _StaticRankFactoredBias)
        for module in deployment.modules()
    ) == 2
    assert deployment.onnx_deployment_inventory["static_shifted_splits"]
    assert deployment.onnx_deployment_inventory["static_ocab_biases"]
    assert not any(parameter.requires_grad for parameter in deployment.parameters())


def test_prepare_for_onnx_export_enforces_static_contract() -> None:
    model = _tiny_drft()

    with pytest.raises(ValueError, match="concrete integer NCHW"):
        model.prepare_for_onnx_export((1, 3, None, 8))
    with pytest.raises(ValueError, match="at least 2x2"):
        model.prepare_for_onnx_export((1, 3, 4, 8))

    deployment = model.prepare_for_onnx_export((1, 3, 8, 8))
    assert deployment.prepare_for_onnx_export(
        (1, 3, 8, 8), clone=False,
    ) is deployment
    with pytest.raises(RuntimeError, match="static input shape"):
        deployment(torch.rand(1, 3, 12, 8))
    with pytest.raises(RuntimeError, match="already prepared"):
        deployment.prepare_for_onnx_export((1, 3, 12, 8), clone=False)


def test_prepare_for_onnx_export_supports_one_dynamic_spatial_graph() -> None:
    torch.manual_seed(7)
    model = _tiny_drft()
    deployment = model.prepare_for_onnx_export(
        (1, 3, 8, 8), dynamic_spatial=True,
    )

    assert deployment._onnx_deployment_dynamic_spatial is True
    assert deployment._onnx_deployment_feature_shape is None
    assert deployment.onnx_deployment_inventory["example_shape"] == (1, 3, 8, 8)
    assert deployment.onnx_deployment_inventory["feature_shape"] == "dynamic"
    assert deployment.onnx_deployment_inventory["attention_export"] == "explicit_math"
    assert deployment.onnx_deployment_inventory["q4r_route"] == "split4"
    assert all(
        module.force_math_attention
        for module in deployment.modules()
        if isinstance(module, (WindowAttentionRFB, NarrowWindowAttention, OCAB))
    )

    for shape in ((1, 3, 8, 8), (1, 3, 12, 8), (1, 3, 8, 12)):
        sample = torch.rand(shape)
        with torch.inference_mode():
            expected = model(sample)
            actual = deployment(sample)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    assert deployment.prepare_for_onnx_export(
        (1, 3, 12, 12), clone=False, dynamic_spatial=True,
    ) is deployment
    with pytest.raises(RuntimeError, match="fixed N/C with dynamic H/W"):
        deployment(torch.rand(2, 3, 8, 8))


def test_prepare_for_onnx_export_owns_tensorrt_mixed_precision() -> None:
    model = _tiny_drft()
    deployment = model.prepare_for_onnx_export(
        (1, 3, 8, 8), precision="tensorrt_mixed",
    )

    assert isinstance(deployment, _DRFTTensorRTMixedPrecision)
    assert deployment.onnx_deployment_precision_handled is True
    assert deployment.onnx_deployment_inventory["io_dtype"] == "torch.float32"
    assert deployment.onnx_deployment_inventory["body_dtype"] == "torch.bfloat16"
    assert deployment.onnx_deployment_inventory["fp16_regions"] == [
        "upsample", "conv_last",
    ]
    assert {
        parameter.dtype for parameter in deployment.inner.upsample.parameters()
    } == {torch.float16}
    assert {
        parameter.dtype for parameter in deployment.inner.conv_last.parameters()
    } == {torch.float16}
    assert deployment.inner.conv_first.weight.dtype is torch.bfloat16

    dynamic_deployment = model.prepare_for_onnx_export(
        (1, 3, 8, 8),
        precision="tensorrt_mixed",
        dynamic_spatial=True,
    )
    assert isinstance(dynamic_deployment, _DRFTTensorRTMixedPrecision)
    assert dynamic_deployment.inner._onnx_deployment_dynamic_spatial is True

    with pytest.raises(ValueError, match="unsupported DRFT ONNX precision"):
        model.prepare_for_onnx_export((1, 3, 8, 8), precision="fp8")
