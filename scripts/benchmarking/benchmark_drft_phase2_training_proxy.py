#!/usr/bin/env python3
# ruff: noqa: E402, I001, SLF001
"""Measure the frozen Phase-2 DRFT training-step proxy.

This is a compute-step benchmark with resident synthetic tensors.  It includes
forward, Charbonnier loss, backward, AdamW update, and EMA update, but excludes
data loading and validation.  Run separate processes for repeated measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from ema_pytorch import EMA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traiNNer.archs.drft_arch import ImageLayerNorm, drft_micro, drft_xs
from traiNNer.losses.basic_loss import CharbonnierLoss
from traiNNer.utils.compile_cache import configure_compile_cache_runtime


FACTORIES = {
    "drft_micro": drft_micro,
    "drft_xs": drft_xs,
}


def _native_image_layer_norm_forward(
    module: ImageLayerNorm, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental whole-image native LayerNorm used only by this harness."""
    normalized, _mean, reciprocal_std = torch.native_layer_norm(
        x,
        (x.shape[1], x.shape[2]),
        None,
        None,
        module.eps,
    )
    std = reciprocal_std.reciprocal().reshape(x.shape[0], 1, 1)
    return normalized * module.weight + module.bias, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(FACTORIES), default=list(FACTORIES))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument(
        "--fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require the entire DRFT forward to compile as one Dynamo graph.",
    )
    parser.add_argument(
        "--dynamic",
        choices=("auto", "static", "dynamic"),
        default="auto",
        help="Shape policy passed to torch.compile.",
    )
    parser.add_argument(
        "--inductor-option",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Repeatable torch.compile Inductor option, for example combo_kernels=true.",
    )
    parser.add_argument(
        "--finalize-precompile-cache",
        action="store_true",
        help=(
            "After warmup, persist PyTorch's experimental Dynamo precompile cache. "
            "Requires TORCH_CACHING_PRECOMPILE=1 before Python starts."
        ),
    )
    parser.add_argument(
        "--attn-type",
        choices=("masked", "hybrid"),
        default="masked",
        help="DRFT shifted-window attention route to benchmark.",
    )
    parser.add_argument(
        "--iln-implementation",
        choices=("manual", "native"),
        default="manual",
        help=(
            "Select the production paper i-LN or an experimental native "
            "LayerNorm lowering. The native route is benchmark-only."
        ),
    )
    parser.add_argument(
        "--prepare-for-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prime DRFT's shape-dependent caches before torch.compile.",
    )
    parser.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ACT activation checkpointing (the current DRFT training default).",
    )
    parser.add_argument(
        "--checkpoint-ocab",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable OCAB activation checkpointing.",
    )
    parser.add_argument(
        "--ema-mode",
        choices=("scalar", "foreach", "off"),
        default="scalar",
        help="EMA implementation to benchmark.",
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=("default", "foreach", "fused"),
        default="default",
        help="AdamW implementation to benchmark.",
    )
    parser.add_argument(
        "--include-grad-norm",
        action="store_true",
        help="Include the real trainer's per-parameter gradient-norm calculation.",
    )
    parser.add_argument(
        "--include-finite-check",
        action="store_true",
        help="Include the real trainer's Python loss-finiteness branch.",
    )
    parser.add_argument(
        "--retain-output",
        action="store_true",
        help="Keep the prior prediction alive across steps like SRModel.self.output.",
    )
    parser.add_argument(
        "--cudagraph-mark-step-begin",
        action="store_true",
        help="Explicitly mark each training iteration for CUDA Graph Trees.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "Bracket only the measured steady-state iterations with the CUDA "
            "Profiler API for Nsight Systems capture-range profiling."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_inductor_options(values: list[str]) -> dict[str, object]:
    options: dict[str, object] = {}
    for item in values:
        key, separator, raw_value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid --inductor-option {item!r}; expected KEY=JSON_VALUE")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        options[key] = value
    return options


def run_one(name: str, args: argparse.Namespace) -> dict[str, object]:
    run_start = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch._dynamo.config.caching_precompile:
        # PyTorch 2.14's experimental compiled-function serializer records the
        # public SDPBackend enum under its pybind class name.  Supply that
        # missing module alias before artifacts are captured.
        import torch.nn.attention as attention_module
        from torch.nn.attention import SDPBackend

        if not hasattr(attention_module, "_SDPBackend"):
            attention_module._SDPBackend = SDPBackend

    if args.iln_implementation == "native":
        ImageLayerNorm.forward = _native_image_layer_norm_forward

    bare_model = FACTORIES[name](
        scale=4,
        use_checkpoint=args.checkpoint,
        use_checkpoint_ocab=args.checkpoint_ocab,
        window_size=32,
        rank=32,
        attn_type=args.attn_type,
    ).to(device="cuda", memory_format=torch.channels_last)

    parameter_count = sum(parameter.numel() for parameter in bare_model.parameters())
    ema = None
    if args.ema_mode != "off":
        ema = EMA(
            bare_model,
            beta=0.999,
            update_after_step=0,
            update_every=1,
            power=0.75,
            allow_different_devices=True,
            use_foreach=args.ema_mode == "foreach",
        ).to(device="cuda", memory_format=torch.channels_last)
    compile_wrapper_ms = 0.0
    if args.compile:
        architecture_cache_tag = getattr(bare_model, "compile_cache_key_tag", None)
        if architecture_cache_tag:
            iln_cache_suffix = (
                ":iln=native" if args.iln_implementation == "native" else ""
            )
            torch.compiler.config.cache_key_tag = (
                f"{architecture_cache_tag}:mode={args.compile_mode}"
                f":input={args.batch_size}x{args.crop_size}x{args.crop_size}"
                ":amp=True-torch.bfloat16:layout=torch.channels_last"
                ":autotune-at-compile-time="
                f"{torch._inductor.config.triton.autotune_at_compile_time}"
                ":static-triton-launcher="
                f"{torch._inductor.config.use_static_triton_launcher}"
                f"{iln_cache_suffix}"
            )
        configure_compile_cache_runtime(
            torch.compiler.config.cache_key_tag
            or f"benchmark={name}:mode={args.compile_mode}"
        )
        if args.prepare_for_compile:
            bare_model.prepare_for_compile(
                input_shape=(args.batch_size, 3, args.crop_size, args.crop_size),
                input_dtype=torch.bfloat16,
            )
        mode_options = torch._inductor.list_mode_options(args.compile_mode)
        if torch._dynamo.config.caching_precompile:
            # Package restoration happens before the backend processes
            # torch.compile(mode=...), so mirror the requested mode's
            # CUDA-Graph setting for deserialization as the trainer does.
            torch._inductor.config.triton.cudagraphs = bool(
                mode_options.get("triton.cudagraphs", False)
            )
        compile_kwargs: dict[str, object] = {"fullgraph": args.fullgraph}
        if args.dynamic != "auto":
            compile_kwargs["dynamic"] = args.dynamic == "dynamic"
        inductor_options = parse_inductor_options(args.inductor_option)
        if inductor_options:
            compile_options = mode_options
            compile_options.update(inductor_options)
            compile_kwargs["options"] = compile_options
        else:
            compile_kwargs["mode"] = args.compile_mode
        compile_start = time.perf_counter()
        model = torch.compile(bare_model, **compile_kwargs)
        compile_wrapper_ms = (time.perf_counter() - compile_start) * 1000.0
    else:
        model = bare_model
    model.train()
    optimizer_options: dict[str, object] = {}
    if args.optimizer_mode == "foreach":
        optimizer_options["foreach"] = True
    elif args.optimizer_mode == "fused":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        **optimizer_options,
    )
    criterion = CharbonnierLoss(loss_weight=1.0)

    lq = torch.rand(
        args.batch_size,
        3,
        args.crop_size,
        args.crop_size,
        device="cuda",
    ).contiguous(memory_format=torch.channels_last)
    gt = torch.rand(
        args.batch_size,
        3,
        args.crop_size * 4,
        args.crop_size * 4,
        device="cuda",
    ).contiguous(memory_format=torch.channels_last)

    retained_prediction: torch.Tensor | None = None

    def step() -> None:
        nonlocal retained_prediction
        if args.cudagraph_mark_step_begin:
            torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(lq)
            loss = criterion(prediction, gt)
        if args.retain_output:
            retained_prediction = prediction
        if args.include_finite_check and not loss.isfinite():
            raise RuntimeError("non-finite benchmark loss")
        loss.backward()
        if args.include_grad_norm:
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            torch.linalg.vector_norm(
                torch.cat([gradient.reshape(-1) for gradient in gradients]), 2
            )
        optimizer.step()
        if ema is not None:
            ema.update()

    setup_ms = (time.perf_counter() - run_start) * 1000.0
    torch.cuda.reset_peak_memory_stats()
    warmup_ms: list[float] = []
    for _ in range(args.warmup):
        warmup_start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        warmup_ms.append((time.perf_counter() - warmup_start) * 1000.0)
    torch.cuda.synchronize()

    precompile_cache_save_ms: float | None = None
    precompile_cache_info: dict[str, object] | None = None
    if args.finalize_precompile_cache:
        if not args.compile:
            raise ValueError("--finalize-precompile-cache requires --compile")
        if not torch._dynamo.config.caching_precompile:
            raise RuntimeError(
                "--finalize-precompile-cache requires TORCH_CACHING_PRECOMPILE=1 "
                "before Python starts"
            )
        from torch._dynamo.precompile_context import PrecompileContext

        cache_save_start = time.perf_counter()
        precompile_cache_info = PrecompileContext.save_to_dynamo_cache()
        precompile_cache_save_ms = (
            time.perf_counter() - cache_save_start
        ) * 1000.0

    step_ms: list[float] = []
    allocated_bytes: list[int] = []
    reserved_bytes: list[int] = []
    if args.cuda_profiler_range:
        torch.cuda.profiler.start()
    try:
        for _ in range(args.steps):
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - start) * 1000.0)
            allocated_bytes.append(torch.cuda.memory_allocated())
            reserved_bytes.append(torch.cuda.memory_reserved())
    finally:
        if args.cuda_profiler_range:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()

    mean_ms = statistics.fmean(step_ms)
    result: dict[str, object] = {
        "model": name,
        "parameters": parameter_count,
        "setup_ms": setup_ms,
        "compile_wrapper_ms": compile_wrapper_ms,
        "warmup_ms": warmup_ms,
        "precompile_cache_save_ms": precompile_cache_save_ms,
        "precompile_cache_info": precompile_cache_info,
        "step_ms": step_ms,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(step_ms),
        "p95_ms": percentile(step_ms, 0.95),
        "cv": statistics.pstdev(step_ms) / mean_ms if len(step_ms) > 1 else 0.0,
        "images_per_second": args.batch_size * 1000.0 / mean_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "end_allocated_bytes": allocated_bytes[-1],
        "end_reserved_bytes": reserved_bytes[-1],
        "allocated_bytes_by_step": allocated_bytes,
        "reserved_bytes_by_step": reserved_bytes,
    }

    del criterion, optimizer, model, ema, bare_model, lq, gt
    torch.cuda.empty_cache()
    if args.compile:
        torch._dynamo.reset()
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 1 or args.steps < 1:
        raise ValueError("--warmup and --steps must both be positive")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    arch_path = REPO_ROOT / "traiNNer" / "archs" / "drft_arch.py"
    payload: dict[str, object] = {
        "protocol": {
            "scale": 4,
            "batch_size": args.batch_size,
            "lr_crop": args.crop_size,
            "dtype": "bf16_autocast",
            "memory_format": "channels_last",
            "loss": "CharbonnierLoss(eps=1e-12)",
            "optimizer": f"AdamW(lr=2e-4, betas=(0.9,0.99), weight_decay=0,{args.optimizer_mode=})",
            "ema": f"beta=0.999,power=0.75,update_every=1,{args.ema_mode=}",
            "window_size": 32,
            "rank": 32,
        "attn_type": args.attn_type,
        "iln_implementation": args.iln_implementation,
            "use_checkpoint": args.checkpoint,
            "use_checkpoint_ocab": args.checkpoint_ocab,
            "include_grad_norm": args.include_grad_norm,
            "include_finite_check": args.include_finite_check,
            "retain_output": args.retain_output,
            "cudagraph_mark_step_begin": args.cudagraph_mark_step_begin,
            "cuda_profiler_range": args.cuda_profiler_range,
            "compiled": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "fullgraph": args.fullgraph,
            "dynamic": args.dynamic,
            "inductor_options": parse_inductor_options(args.inductor_option),
            "finalize_precompile_cache": args.finalize_precompile_cache,
            "prepare_for_compile": args.prepare_for_compile,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "data_loading_included": False,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "drft_arch_sha256": hashlib.sha256(arch_path.read_bytes()).hexdigest(),
        },
        "results": [],
    }
    results = [run_one(name, args) for name in args.models]
    payload["results"] = results

    by_name = {str(result["model"]): result for result in results}
    if "drft_micro" in by_name and "drft_xs" in by_name:
        payload["micro_speedup_vs_xs"] = (
            float(by_name["drft_xs"]["mean_ms"])
            / float(by_name["drft_micro"]["mean_ms"])
        )

    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
