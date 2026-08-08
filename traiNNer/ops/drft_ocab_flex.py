# ruff: noqa: SLF001
"""Fused training-only OCAB attention for compiled DRFT models.

The public Flex Attention higher-order op is fast, but PyTorch 2.14's
experimental Dynamo package serializer cannot persist a full model graph that
contains it after Triton launchers have materialized.  This module gives the
outer DRFT graph a stable custom-op boundary while retaining a raw AOTAutograd
forward/backward kernel pair behind that boundary.  The raw pair avoids nesting
an autograd engine inside the outer compiled backward, which would otherwise
force expensive CUDA Graph recording again on every fresh process.

Nothing in this module participates in eager inference or ONNX/TensorRT export.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

_KernelKey = tuple[Any, ...]
_DirectFlexCallable = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
_PreparedFlexForward = Callable[
    [Tensor, Tensor, Tensor, Tensor], tuple[Tensor, tuple[Tensor, ...]]
]
_PreparedFlexBackward = Callable[
    [tuple[Tensor, ...], Tensor], tuple[Tensor, Tensor, Tensor, Tensor]
]
_ScoreMod = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class _PreparedFlexKernel:
    forward_with_saved: _PreparedFlexForward
    backward: _PreparedFlexBackward


_PREPARED_KERNELS: dict[_KernelKey, _PreparedFlexKernel] = {}
_PREPARE_LOCK = threading.RLock()


def _kernel_key(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias_flat: Tensor,
    scale: float,
) -> _KernelKey:
    tensors = (q, k, v, bias_flat)
    return (
        q.device.type,
        q.device.index,
        *(str(tensor.dtype) for tensor in tensors),
        *(tuple(tensor.shape) for tensor in tensors),
        *(tuple(tensor.stride()) for tensor in tensors),
        float(scale),
    )


def _bias_score_mod(bias_flat: Tensor, seq_len: int) -> _ScoreMod:
    def score_mod(
        score: Tensor,
        b: Tensor,
        h: Tensor,
        q_idx: Tensor,
        kv_idx: Tensor,
    ) -> Tensor:
        del b
        return score + bias_flat[h, q_idx * seq_len + kv_idx]

    return score_mod


def _make_flex_graph(scale: float) -> _DirectFlexCallable:
    def flex_graph(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias_flat: Tensor,
    ) -> Tensor:
        from torch.nn.attention.flex_attention import flex_attention

        return flex_attention(
            q,
            k,
            v,
            score_mod=_bias_score_mod(bias_flat, k.shape[-2]),
            scale=scale,
        )

    return flex_graph


def _require_kernel_support() -> None:
    required = {
        "torch.library.custom_op": hasattr(torch.library, "custom_op"),
    }
    missing = [name for name, available in required.items() if not available]
    if missing:
        raise RuntimeError(
            "Compiled DRFT OCAB fusion is incompatible with torch "
            f"{torch.__version__}; missing {', '.join(missing)}"
        )


def prepare_ocab_flex_training_kernel(
    *,
    batch_windows: int,
    num_heads: int,
    query_tokens: int,
    key_tokens: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    scale: float,
) -> bool:
    """Lower one exact OCAB training shape before compiling the outer model.

    Returns ``True`` when a new pair of kernels was prepared and ``False`` when
    this process already has an identical pair.
    """
    _require_kernel_support()
    if device.type != "cuda":
        raise RuntimeError("Compiled DRFT OCAB fusion requires a CUDA device")

    q = torch.zeros(
        batch_windows,
        query_tokens,
        num_heads,
        head_dim,
        dtype=dtype,
        device=device,
    ).permute(0, 2, 1, 3)
    k = torch.zeros(
        batch_windows,
        key_tokens,
        num_heads,
        head_dim,
        dtype=dtype,
        device=device,
    ).permute(0, 2, 1, 3)
    v = torch.zeros_like(k)
    bias_flat = torch.zeros(
        num_heads,
        query_tokens * key_tokens,
        dtype=dtype,
        device=device,
    )
    key = _kernel_key(q, k, v, bias_flat, scale)

    with _PREPARE_LOCK:
        if key in _PREPARED_KERNELS:
            return False

        try:
            from torch._functorch.aot_autograd import aot_function
            from torch._inductor.compile_fx import compile_fx_inner
            from torch._inductor.decomposition import select_decomp_table
            from torch._inductor.utils import BoxedBool
        except ImportError as error:
            raise RuntimeError(
                "Compiled DRFT OCAB fusion is incompatible with torch "
                f"{torch.__version__}; raw AOTAutograd compiler APIs are missing"
            ) from error

        holders: dict[str, Any] = {}
        config_patches = torch._inductor.list_mode_options(
            "max-autotune-no-cudagraphs"
        )

        def forward_compiler(
            graph_module: torch.fx.GraphModule,
            example_inputs: list[Tensor],
        ) -> Callable[..., Any]:
            with torch._inductor.config.patch(config_patches):
                compiled = compile_fx_inner(
                    graph_module,
                    example_inputs,
                    cudagraphs=BoxedBool(False),
                    is_backward=False,
                )
            holders["forward"] = compiled
            return compiled

        def backward_compiler(
            graph_module: torch.fx.GraphModule,
            example_inputs: list[Tensor],
        ) -> Callable[..., Any]:
            with torch._inductor.config.patch(config_patches):
                compiled = compile_fx_inner(
                    graph_module,
                    example_inputs,
                    cudagraphs=BoxedBool(False),
                    is_backward=True,
                )
            holders["backward"] = compiled
            return compiled

        caching_precompile_available = hasattr(
            torch._dynamo.config, "caching_precompile"
        )
        previous_caching_precompile = (
            torch._dynamo.config.caching_precompile
            if caching_precompile_available
            else None
        )
        if caching_precompile_available:
            # The outer custom op is serializable; Flex's materialized Triton
            # launchers are not. Keep only the stable outer graph in the Dynamo
            # package while the ordinary persistent Inductor/Triton caches own
            # this raw AOTAutograd kernel pair.
            torch._dynamo.config.caching_precompile = False

        try:
            compiled_flex = aot_function(
                _make_flex_graph(scale),
                fw_compiler=forward_compiler,
                bw_compiler=backward_compiler,
                decompositions=select_decomp_table(),
            )

            with torch.enable_grad():
                q_grad = q.detach().requires_grad_(True)
                k_grad = k.detach().requires_grad_(True)
                v_grad = v.detach().requires_grad_(True)
                bias_grad = bias_flat.detach().requires_grad_(True)
                output = compiled_flex(
                    q_grad,
                    k_grad,
                    v_grad,
                    bias_grad,
                )
                torch.autograd.grad(
                    output,
                    (q_grad, k_grad, v_grad, bias_grad),
                    torch.ones_like(output),
                )
            torch.cuda.synchronize(device)
        finally:
            if caching_precompile_available:
                torch._dynamo.config.caching_precompile = (
                    previous_caching_precompile
                )

        try:
            raw_forward = holders["forward"]
            raw_backward = holders["backward"]
        except KeyError as error:
            raise RuntimeError(
                "Inductor did not publish the raw DRFT OCAB kernel pair"
            ) from error

        def forward_with_saved(
            q_input: Tensor,
            k_input: Tensor,
            v_input: Tensor,
            bias_input: Tensor,
        ) -> tuple[Tensor, tuple[Tensor, ...]]:
            outputs = raw_forward([q_input, k_input, v_input, bias_input])
            if isinstance(outputs, Tensor):
                outputs = [outputs]
            if not isinstance(outputs, list | tuple) or not outputs:
                raise RuntimeError(
                    "Raw DRFT OCAB forward returned an invalid output sequence"
                )
            if not all(isinstance(tensor, Tensor) for tensor in outputs):
                raise RuntimeError(
                    "Raw DRFT OCAB forward returned a non-tensor output"
                )
            return outputs[0], tuple(outputs[1:])

        def backward(
            saved_tensors: tuple[Tensor, ...],
            grad_output: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            outputs = raw_backward([*saved_tensors, grad_output])
            if isinstance(outputs, Tensor):
                outputs = [outputs]
            if not isinstance(outputs, list | tuple) or len(outputs) != 4:
                raise RuntimeError(
                    "Raw DRFT OCAB backward did not return four gradients"
                )
            if not all(isinstance(tensor, Tensor) for tensor in outputs):
                raise RuntimeError(
                    "Raw DRFT OCAB backward returned a non-tensor gradient"
                )
            return outputs[0], outputs[1], outputs[2], outputs[3]

        prepared = _PreparedFlexKernel(
            forward_with_saved=forward_with_saved,
            backward=backward,
        )
        _PREPARED_KERNELS[key] = prepared
        return True


def _prepared_kernel(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias_flat: Tensor,
    scale: float,
) -> _PreparedFlexKernel:
    key = _kernel_key(q, k, v, bias_flat, scale)
    try:
        return _PREPARED_KERNELS[key]
    except KeyError as error:
        raise RuntimeError(
            "DRFT's fused OCAB training kernel was not prepared for input "
            f"shapes {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        ) from error


@torch.library.custom_op(
    "trainner_drft::ocab_flex_forward",
    mutates_args=(),
    device_types="cuda",
)
def _ocab_flex_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias_flat: Tensor,
    scale: float,
) -> Tensor:
    output, _ = _prepared_kernel(
        q, k, v, bias_flat, scale
    ).forward_with_saved(q, k, v, bias_flat)
    return output


@_ocab_flex_forward.register_fake
def _ocab_flex_forward_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias_flat: Tensor,
    scale: float,
) -> Tensor:
    del k, v, bias_flat, scale
    return torch.empty_like(q)


@torch.library.custom_op(
    "trainner_drft::ocab_flex_backward",
    mutates_args=(),
    device_types="cuda",
)
def _ocab_flex_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias_flat: Tensor,
    grad_output: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    kernel = _prepared_kernel(q, k, v, bias_flat, scale)
    _, saved_tensors = kernel.forward_with_saved(q, k, v, bias_flat)
    return kernel.backward(saved_tensors, grad_output)


@_ocab_flex_backward.register_fake
def _ocab_flex_backward_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias_flat: Tensor,
    grad_output: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del grad_output, scale
    return tuple(torch.empty_like(tensor) for tensor in (q, k, v, bias_flat))


def _setup_context(
    ctx: Any,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, float],
    output: Tensor,
) -> None:
    del output
    q, k, v, bias_flat, scale = inputs
    ctx.save_for_backward(q, k, v, bias_flat)
    ctx.scale = scale


def _backward(
    ctx: Any,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
    q, k, v, bias_flat = ctx.saved_tensors
    gradients = torch.ops.trainner_drft.ocab_flex_backward.default(
        q,
        k,
        v,
        bias_flat,
        grad_output,
        ctx.scale,
    )
    return (*gradients, None)


torch.library.register_autograd(
    _ocab_flex_forward,
    _backward,
    setup_context=_setup_context,
)

ocab_flex_training_op = torch.ops.trainner_drft.ocab_flex_forward.default
