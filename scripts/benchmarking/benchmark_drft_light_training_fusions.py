# ruff: noqa: I001
"""Reproducible compiled DRFT-Light training-step benchmark.

This deliberately measures the model-side training hot path only: generator
forward, Charbonnier loss, backward, and fused AdamW update. Data loading,
degradation, validation, logging, guards, and EMA are outside the timed region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from traiNNer.archs.drft_arch import AttentionBlocks, OCAB, drft_light


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "traiNNer" / "archs" / "drft_arch.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument(
        "--cuda-profiler",
        action="store_true",
        help="Bracket measured steps with the CUDA profiler API for Nsight Systems.",
    )
    parser.add_argument(
        "--torch-profile-output",
        type=Path,
        help="Write a PyTorch CPU/CUDA Chrome trace and a sibling kernel table.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def gpu_snapshot() -> dict[str, Any]:
    query = (
        "name,driver_version,memory.total,memory.used,temperature.gpu,"
        "clocks.current.graphics,clocks.current.memory,power.draw"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()}
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    keys = [
        "name",
        "driver_version",
        "memory_total_mib",
        "memory_used_mib",
        "temperature_c",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "power_w",
    ]
    return dict(zip(keys, values, strict=True))


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = prediction - target
    return torch.sqrt(difference * difference + 1e-12).mean()


def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    low_resolution: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(low_resolution)
        loss = charbonnier_loss(prediction, ground_truth)
    loss.backward()
    optimizer.step()
    return loss.detach()


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.batch_size, args.height, args.width, args.warmup, args.runs) <= 0:
        raise ValueError("batch/shape/warmup/runs must all be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch._dynamo.config.specialize_float = True  # noqa: SLF001

    input_shape = (args.batch_size, 3, args.height, args.width)
    output_shape = (
        args.batch_size,
        3,
        args.height * 4,
        args.width * 4,
    )
    device = torch.device("cuda")
    memory_format = torch.channels_last

    model = drft_light(
        scale=4,
        window_size=32,
        overlap_window_size=48,
        rank=32,
        drop_path_rate=0.1,
        use_checkpoint=True,
        use_checkpoint_ocab=False,
        attn_type="hybrid",
    ).train()
    model = model.to(device=device, memory_format=memory_format)
    model.prepare_for_compile(input_shape=input_shape, input_dtype=torch.bfloat16)
    attention_groups = [
        module for module in model.modules() if isinstance(module, AttentionBlocks)
    ]
    ocab_modules = [module for module in model.modules() if isinstance(module, OCAB)]
    automatic_fusion_inventory = {
        "attention_groups": len(attention_groups),
        "selective_checkpoint_groups": sum(
            module.use_compiled_checkpoint_policy for module in attention_groups
        ),
        "ocab_modules": len(ocab_modules),
        "ocab_batch_axis_kv": sum(
            module._use_batch_axis_kv  # noqa: SLF001
            for module in ocab_modules
        ),
        "ocab_compact_relative_bias": sum(
            module.use_compact_relative_bias_training for module in ocab_modules
        ),
        "ocab_compact_unfold": sum(
            module.use_compact_unfold_training for module in ocab_modules
        ),
        "ocab_fused_qkv_windows": sum(
            module.use_fused_qkv_windows_training for module in ocab_modules
        ),
        "ocab_window_order_projection": sum(
            module.use_window_order_projection_training for module in ocab_modules
        ),
        "ocab_attention": "sdpa",
    }
    compiled_model = torch.compile(model, mode=args.compile_mode)

    optimizer = torch.optim.AdamW(
        compiled_model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        fused=True,
    )
    low_resolution = torch.rand(input_shape, device=device, dtype=torch.float32)
    ground_truth = torch.rand(output_shape, device=device, dtype=torch.float32)
    low_resolution = low_resolution.to(memory_format=memory_format)
    ground_truth = ground_truth.to(memory_format=memory_format)

    start_gpu = gpu_snapshot()
    compile_start = time.perf_counter()
    warmup_losses: list[float] = []
    for _ in range(args.warmup):
        loss = training_step(compiled_model, optimizer, low_resolution, ground_truth)
        torch.cuda.synchronize()
        warmup_losses.append(float(loss))
    compile_and_warmup_seconds = time.perf_counter() - compile_start

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings_ms: list[float] = []
    measured_losses: list[float] = []

    profile_context = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        if args.torch_profile_output is not None
        else nullcontext()
    )
    with profile_context as active_profile:
        if args.cuda_profiler:
            torch.cuda.profiler.start()
        try:
            for _ in range(args.runs):
                start_event.record(stream)
                loss = training_step(
                    compiled_model, optimizer, low_resolution, ground_truth
                )
                end_event.record(stream)
                end_event.synchronize()
                timings_ms.append(float(start_event.elapsed_time(end_event)))
                measured_losses.append(float(loss))
        finally:
            if args.cuda_profiler:
                torch.cuda.profiler.stop()
    if args.torch_profile_output is not None:
        args.torch_profile_output.parent.mkdir(parents=True, exist_ok=True)
        active_profile.export_chrome_trace(str(args.torch_profile_output))
        profile_table = active_profile.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=100,
        )
        args.torch_profile_output.with_suffix(".txt").write_text(profile_table + "\n")
    torch.cuda.synchronize()

    mean_ms = statistics.fmean(timings_ms)
    result = {
        "schema": "drft-light-training-fusion-benchmark-v1",
        "label": args.label,
        "source": str(SOURCE_PATH),
        "source_sha256": sha256_file(SOURCE_PATH),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "compile_mode": args.compile_mode,
            "deterministic": True,
            "channels_last": True,
            "autocast": "bfloat16",
            "gpu_start": start_gpu,
            "gpu_end": gpu_snapshot(),
        },
        "recipe": {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "warmup": args.warmup,
            "runs": args.runs,
            "seed": args.seed,
            "optimizer": "fused AdamW",
            "learning_rate": 2e-4,
            "betas": [0.9, 0.99],
            "weight_decay": 0.0,
            "loss": "Charbonnier eps=1e-12 mean",
            "act_checkpointing": True,
            "ocab_checkpointing": False,
            "attention": "hybrid",
            "window_size": 32,
            "ocab_span": 48,
            "rank": 32,
            "drop_path_rate": 0.1,
            "training_fusions": "automatic",
            "automatic_fusion_inventory": automatic_fusion_inventory,
        },
        "compile_and_warmup_seconds": compile_and_warmup_seconds,
        "timings_ms": timings_ms,
        "summary": {
            "mean_ms": mean_ms,
            "p50_ms": percentile(timings_ms, 50),
            "p95_ms": percentile(timings_ms, 95),
            "min_ms": min(timings_ms),
            "max_ms": max(timings_ms),
            "iterations_per_second": 1000.0 / mean_ms,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            "warmup_last_loss": warmup_losses[-1],
            "measured_last_loss": measured_losses[-1],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2, allow_nan=False))
    print(f"result={args.output}")


if __name__ == "__main__":
    main()
