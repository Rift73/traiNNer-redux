"""Profile the real OTF degradation path (RealESRGANModel.feed_data), per stage.

The script builds the same objects train.py builds: the options through
Config.load_config_from_file (parse_options), the train DataLoader through
train.create_train_val_dataloader, the CUDAPrefetcher, and the model through
build_model. It then times feed_data on real batches. No training step runs.

After the warm-up iterations, passes A and B alternate batch by batch, so both
see the same drift (queue fill, caches, clocks):
  A. feed_data alone, with torch.cuda.synchronize() before and after each call
     and no synchronisation inside it (the "unsynced" total). The stage
     wrappers stay installed but inactive (one extra Python call per stage).
  B. the same, with every OTF stage wrapped by a timer that synchronises CUDA
     before and after the stage. Those extra syncs serialise CPU and GPU and
     remove any overlap between stages, so the pass-B total and the summed
     stage times are normally larger than the pass-A total; the report
     prints both.
Then C, a short sync census: torch.cuda.set_sync_debug_mode("warn") inside
feed_data only, counting the implicit host syncs per source line.
With --train-steps N, D: real training iterations on the same model, run the
way train.py runs one iteration (feed_data, optimize_parameters,
finalize_compile_cache, update_learning_rate, with the template's optimizer,
loss, EMA, AMP and torch.compile settings). The first --warmup steps carry
the compile and CUDA graph capture; they are reported separately and excluded
from the step statistics.

The dataloader fetch time is CUDAPrefetcher.next() plus a synchronize, so it
includes the wait for the workers and the host-to-device copy of the batch.

Run from the work tree root (Linux/WSL):
  CUDA_HOME=/usr/local/cuda-13.2 PYTHONPATH=$PWD \\
    /home/rift/workspace/traiNNer-redux/venv/bin/python benchmarks/otf_profile.py
"""

from __future__ import annotations

import argparse
import linecache
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from os import path as osp
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import msgspec
import numpy as np
import torch
import traiNNer.models.realesrgan_model as realesrgan_model_module
from train import create_train_val_dataloader
from traiNNer.data.prefetch_dataloader import CUDAPrefetcher
from traiNNer.models import build_model
from traiNNer.models.base_model import BaseModel
from traiNNer.models.realesrgan_model import RealESRGANModel
from traiNNer.utils import get_root_logger
from traiNNer.utils.config import Config
from traiNNer.utils.misc import set_random_seed
from traiNNer.utils.redux_options import ReduxOptions

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
DEFAULT_OPT = osp.join(ROOT, "options", "train", "DRFT", "4x_DRFT_S.yml")
DEFAULT_DATAROOT = "/home/rift/workspace/traiNNer-redux/datasets/val/MANGA109/HR"
DEFAULT_OUT = osp.join(ROOT, "benchmarks", "otf_profile_before.txt")

CUSTOM_PROBS = (
    "ntsc_prob",
    "rainbow_prob",
    "ghosting_prob",
    "interlace_prob",
    "scanline_prob",
    "lowpass_prob",
    "overshoot_prob",
    "shift_prob",
    "subsampling_prob",
    "dithering_prob",
    "otf_hf_noise_prob",
)

# OTF_documentation.md section 4, "Mixed compression with JPEG, WebP, and HEVC".
# The JPEG quality ranges are set in apply_overrides for both codec choices.
MIXED_CODECS: dict[str, Any] = {
    "compress_algorithms": ["jpeg", "hevc", "webp"],
    "compress_algorithm_probs": [0.5, 0.3, 0.2],
    "compress_hevc_range": (20, 40),
    "compress_webp_range": (50, 95),
    "compress_video_sampling": ["444", "422", "420"],
    "compress_algorithms2": ["jpeg", "webp"],
    "compress_algorithm_probs2": [0.7, 0.3],
    "compress_webp_range2": (60, 95),
}

# Model attribute -> stage label. jpeger is the DiffJPEG module instance.
MODEL_STAGES = {
    "_select_active_degradations": "_select_active_degradations",
    "_apply_ntsc": "_apply_ntsc",
    "_apply_rainbow": "_apply_rainbow",
    "_apply_ghosting": "_apply_ghosting",
    "_apply_interlace": "_apply_interlace",
    "_apply_scanline": "_apply_scanline",
    "_apply_lowpass": "_apply_lowpass",
    "_apply_overshoot": "_apply_overshoot",
    "_apply_shift": "_apply_shift",
    "_apply_subsampling": "_apply_subsampling",
    "_apply_compression": "_apply_compression",
    "_apply_hf_noise": "_apply_hf_noise",
    "_apply_dithering": "_apply_dithering",
    "_dequeue_and_enqueue": "_dequeue_and_enqueue",
    "jpeger": "jpeger (DiffJPEG)",
}

# Names that feed_data looks up in the realesrgan_model module namespace.
MODULE_FUNCTIONS = (
    "nlmeans_denoise_pt",
    "filter2d",
    "resize_pt",
    "random_add_gaussian_noise_pt",
    "random_add_poisson_noise_pt",
    "paired_random_crop",
    "paired_random_crop_multi_gt",
    "apply_per_image",
    "apply_per_image_parallel",
    "compress_video_batch",
)

# Where each stage runs, from reading the code paths that this config executes.
STAGE_WHERE = {
    "_select_active_degradations": "CPU (Python only)",
    "_apply_compression": "dispatch (children below)",
    "_apply_dithering": "dispatch (children below)",
    "_apply_ntsc": "GPU + .item() sync (FIR cache key)",
    "_apply_hf_noise": "GPU + .item() sync (gray mask)",
    "_dequeue_and_enqueue": "GPU (CPU randperm index)",
    "apply_per_image": "CPU: D2H, serial numpy per image, H2D",
    "apply_per_image_parallel": "CPU: D2H, 8-thread pool, H2D",
    "compress_video_batch": "CPU: D2H, TorchCodec/PyAV encode+decode, H2D",
}
OFF_GPU_STAGES = ("apply_per_image", "apply_per_image_parallel", "compress_video_batch")

OTF_OPTION_PREFIXES = (
    "high_order_degradation",
    "scale",
    "queue_size",
    "use_amp",
    "use_channels_last",
    "lq_usm",
    "thicklines",
    "blur_",
    "resize_",
    "gaussian_noise",
    "noise_range",
    "poisson_scale",
    "gray_noise",
    "compress_",
    "otf_",
    "dithering_",
    "shift_",
    "subsampling_",
    "ghosting_",
    "interlace_",
    "lowpass_",
    "ntsc_",
    "overshoot_",
    "rainbow_",
    "scanline_",
)


class StageTimer:
    """Wall-clock stage timer with CUDA synchronisation before and after each stage.

    Nested stages are attributed exclusively: a parent's self time excludes the
    inclusive time of the stages it called.
    """

    def __init__(self) -> None:
        self.active = False
        self.history: list[dict[str, list[float]]] = []
        self.parents: dict[str, str] = {}
        self._current: dict[str, list[float]] = {}
        self._stack: list[tuple[str, list[float]]] = []

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:
            if not self.active:
                return fn(*args, **kwargs)
            if self._stack:
                self.parents.setdefault(name, self._stack[-1][0])
            child_time = [0.0]
            self._stack.append((name, child_time))
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                self._stack.pop()
                if self._stack:
                    self._stack[-1][1][0] += elapsed
                record = self._current.setdefault(name, [0.0, 0.0, 0])
                record[0] += elapsed
                record[1] += elapsed - child_time[0]
                record[2] += 1

        return timed

    def end_batch(self) -> None:
        self.history.append(self._current)
        self._current = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--opt", default=DEFAULT_OPT, help="training option file")
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gt-size", type=int, default=256)
    parser.add_argument(
        "--degradations",
        choices=("all", "none"),
        default="all",
        help="all = every custom degradation and both compression stages at 1.0; "
        "none = all of them at 0.0 (--dither, --codecs, --jpeg-quality-cap unused)",
    )
    parser.add_argument(
        "--dither",
        default="floydsteinberg",
        help="single entry for dithering_types (e.g. floydsteinberg, riemersma)",
    )
    parser.add_argument(
        "--codecs",
        choices=("jpeg", "mixed"),
        default="jpeg",
        help="jpeg = [jpeg] in both stages; mixed = OTF_documentation.md section 4 mix",
    )
    parser.add_argument(
        "--jpeg-quality-cap",
        type=int,
        default=None,
        help="keep the option file's JPEG ranges, capped at this quality "
        "(default: use the ReduxOptions default ranges)",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="override compile_mode (default: as in the option file)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--sync-census",
        type=int,
        default=10,
        help="iterations for the implicit-sync census (0 disables it)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help="timed training iterations after the profile, after --warmup untimed ones",
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--append", action="store_true", help="append to --out instead of replacing it"
    )
    return parser.parse_args()


def load_options(args: argparse.Namespace) -> tuple[ReduxOptions, argparse.Namespace]:
    """Parse the option file exactly as train.py does (same argparse entry point)."""
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "-opt", args.opt, "--manual_seed", str(args.seed)]
    try:
        return Config.load_config_from_file(ROOT, is_train=True)
    finally:
        sys.argv = saved_argv


def degradation_overrides(
    opt: ReduxOptions, args: argparse.Namespace
) -> dict[str, Any]:
    if args.degradations == "none":
        # Every custom degradation and both compression stages off; the
        # upstream blur/resize/noise/sinc settings stay as in the option file.
        overrides: dict[str, Any] = dict.fromkeys(CUSTOM_PROBS, 0.0)
        overrides["compress_prob"] = 0.0
        overrides["compress_prob2"] = 0.0
        return overrides

    overrides = dict.fromkeys(CUSTOM_PROBS, 1.0)
    overrides["dithering_types"] = [args.dither]
    # Option files may set compress_prob/compress_prob2 to 0, which would skip
    # the compression stage entirely; force it on so the codec paths are measured.
    overrides["compress_prob"] = 1.0
    overrides["compress_prob2"] = 1.0
    # DiffJPEG maps quality 100 to a quantisation factor of 0 and returns NaN,
    # so a JPEG range reaching 100 cannot run. Either cap the option file's
    # ranges, or (default) use the ReduxOptions defaults.
    if args.jpeg_quality_cap is None:
        option_defaults = {
            field.name: field.default for field in msgspec.structs.fields(ReduxOptions)
        }
        overrides["compress_jpeg_range"] = option_defaults["compress_jpeg_range"]
        overrides["compress_jpeg_range2"] = option_defaults["compress_jpeg_range2"]
    else:
        cap = args.jpeg_quality_cap
        overrides["compress_jpeg_range"] = tuple(
            min(quality, cap) for quality in opt.compress_jpeg_range
        )
        overrides["compress_jpeg_range2"] = tuple(
            min(quality, cap) for quality in opt.compress_jpeg_range2
        )
    # NLMeans only runs when denoise_lq is on (it is gated on otf_hf_noise_prob too).
    overrides["otf_hf_noise_denoise_lq"] = True
    if args.codecs == "jpeg":
        overrides["compress_algorithms"] = ["jpeg"]
        overrides["compress_algorithm_probs"] = [1.0]
        overrides["compress_algorithms2"] = ["jpeg"]
        overrides["compress_algorithm_probs2"] = [1.0]
    else:
        overrides.update(MIXED_CODECS)
    return overrides


def apply_overrides(opt: ReduxOptions, args: argparse.Namespace) -> dict[str, str]:
    """Apply the profile overrides; return 'value (option file: old)' per field."""
    overrides = degradation_overrides(opt, args)
    overrides["otf_degradation_combos"] = None
    if args.compile_mode is not None:
        overrides["compile_mode"] = args.compile_mode

    record: dict[str, str] = {}
    for name, value in overrides.items():
        if name not in opt.__struct_fields__:
            raise AttributeError(f"ReduxOptions has no field {name}")
        record[name] = f"{value} (option file: {getattr(opt, name)})"
        setattr(opt, name, value)

    train_opt = opt.datasets["train"]
    dataset_overrides: dict[str, Any] = {
        "dataroot_gt": [args.dataroot],
        "batch_size_per_gpu": args.batch,
        "num_worker_per_gpu": args.workers,
        "gt_size": args.gt_size,
        "lq_size": None,  # create_train_val_dataloader derives it from gt_size
    }
    for name, value in dataset_overrides.items():
        record[f"datasets.train.{name}"] = (
            f"{value} (option file: {getattr(train_opt, name)})"
        )
        setattr(train_opt, name, value)
    return record


def install_stage_timers(timer: StageTimer, model: RealESRGANModel) -> None:
    for attribute, label in MODEL_STAGES.items():
        setattr(model, attribute, timer.wrap(label, getattr(model, attribute)))
    for name in MODULE_FUNCTIONS:
        original = getattr(realesrgan_model_module, name)
        setattr(realesrgan_model_module, name, timer.wrap(name, original))


def fetch_batch(prefetcher: CUDAPrefetcher) -> tuple[Any, float]:
    start = time.perf_counter()
    data = prefetcher.next()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    if data is None:
        raise RuntimeError("The train loader ran out of batches in the first epoch.")
    return data, elapsed_ms


def timed_feed(model: BaseModel, data: Any) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with model.data_feed_context():
        model.feed_data(data)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def warm_up(model: BaseModel, prefetcher: CUDAPrefetcher, iters: int) -> None:
    for _ in range(iters):
        data, _ = fetch_batch(prefetcher)
        timed_feed(model, data)


def run_interleaved(
    model: BaseModel, prefetcher: CUDAPrefetcher, timer: StageTimer, iters: int
) -> tuple[list[float], list[float], list[float]]:
    """Alternate uninstrumented (A) and instrumented (B) batches, iters of each."""
    fetch_ms: list[float] = []
    feed_a: list[float] = []
    feed_b: list[float] = []
    for index in range(2 * iters):
        instrumented = index % 2 == 1
        data, fetch = fetch_batch(prefetcher)
        fetch_ms.append(fetch)
        timer.active = instrumented
        try:
            feed = timed_feed(model, data)
        finally:
            timer.active = False
        if instrumented:
            feed_b.append(feed)
            timer.end_batch()
        else:
            feed_a.append(feed)
    return fetch_ms, feed_a, feed_b


def sync_census(
    model: BaseModel, prefetcher: CUDAPrefetcher, iters: int
) -> dict[tuple[str, int], int]:
    """Count implicit host syncs inside feed_data per source line."""
    counts: dict[tuple[str, int], int] = defaultdict(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(iters):
            data, _ = fetch_batch(prefetcher)
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with model.data_feed_context():
                    model.feed_data(data)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
    mode_switch_file = osp.realpath(torch.cuda.__file__)
    for warning in caught:
        if "synchroniz" not in str(warning.message).lower():
            continue
        # A sync raised outside a Python frame surfaces on the mode-switch call
        # itself; its source line is unknown, so it is not attributed.
        if osp.realpath(warning.filename) == mode_switch_file:
            counts[("<unattributed: reported on set_sync_debug_mode>", 0)] += 1
            continue
        counts[(warning.filename, warning.lineno)] += 1
    return counts


def run_train_steps(
    model: BaseModel,
    prefetcher: CUDAPrefetcher,
    warmup_iter: int,
    warmup: int,
    steps: int,
) -> tuple[list[float], list[tuple[float, float, bool, bool]]]:
    """Training iterations exactly as train.py's loop body runs them.

    Returns the warm-up step times and, per timed iteration,
    (feed_data ms, step ms, applied the optimizer, step rejected).
    """
    current_iter = 0
    current_accum_iter = 0
    warmup_ms: list[float] = []
    timed: list[tuple[float, float, bool, bool]] = []
    # Progress on stdout lets a caller watch the compile warm-up.
    print("[otf_profile] training start", flush=True)
    for index in range(warmup + steps):
        data, _ = fetch_batch(prefetcher)
        current_accum_iter += 1
        if current_accum_iter >= model.accum_iters:
            current_accum_iter = 0
            current_iter += 1
            apply_gradient = True
        else:
            apply_gradient = False
        feed_ms = timed_feed(model, data)
        start = time.perf_counter()
        model.optimize_parameters(current_iter, current_accum_iter, apply_gradient)
        model.finalize_compile_cache()
        if model.reset_accumulation_requested:
            current_accum_iter = 0
        accepted = apply_gradient and model.training_step_accepted
        if apply_gradient and not accepted:
            current_iter -= 1
        if accepted:
            model.update_learning_rate(current_iter, warmup_iter=warmup_iter)
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - start) * 1e3
        if index < warmup:
            warmup_ms.append(step_ms)
            print(
                f"[otf_profile] train warm-up step {index + 1}/{warmup}: "
                f"{step_ms / 1e3:.1f} s",
                flush=True,
            )
        else:
            rejected = not model.training_step_accepted
            timed.append((feed_ms, step_ms, apply_gradient, rejected))
    return warmup_ms, timed


def training_lines(
    model: RealESRGANModel,
    warmup_ms: list[float],
    records: list[tuple[float, float, bool, bool]],
    feed_a: list[float],
    peak_alloc: int,
) -> list[str]:
    opt = model.opt
    step_ms = [record[1] for record in records]
    accumulate_ms = [record[1] for record in records if not record[2]]
    apply_ms = [record[1] for record in records if record[2]]
    feed_ms = [record[0] for record in records]
    rejected = sum(1 for record in records if record[3])
    compile_text = (
        f"torch.compile mode {opt.compile_mode}" if model.use_compile else "no compile"
    )
    amp_text = f"AMP {model.amp_dtype}" if model.use_amp else "AMP off"
    feed_median = pct(feed_a, 50)
    loop_feed_median = pct(feed_ms, 50)
    step_median = pct(step_ms, 50)
    lines = [
        "Training step - model.optimize_parameters per batch, run as train.py runs one",
        "iteration (then finalize_compile_cache, update_learning_rate); sync before/after:",
        f"  network {opt.network_g['type'] if opt.network_g else None}, {compile_text}, "
        f"{amp_text}, memory format {model.memory_format}, "
        f"accum_iter {model.accum_iters}, batch {opt.datasets['train'].batch_size_per_gpu}, "
        f"losses {list(model.losses)}, optimizer {type(model.optimizer_g).__name__}, "
        f"EMA {'on' if model.net_g_ema is not None else 'off'}",
        "  warm-up steps, excluded (compile, autotune, CUDA graph capture): "
        + ", ".join(f"{ms / 1e3:.1f}" for ms in warmup_ms)
        + f" s; total {sum(warmup_ms) / 1e3:.1f} s",
        "  " + describe(f"train step, all {len(step_ms)} timed batches", step_ms),
    ]
    if accumulate_ms:
        lines.append("  " + describe("  accumulate-only batches", accumulate_ms))
    if apply_ms:
        lines.append("  " + describe("  optimizer-step batches (+EMA, LR)", apply_ms))
    lines += [
        "  " + describe("feed_data inside the training loop", feed_ms),
        f"  rejected (non-finite) steps: {rejected}",
        f"  peak allocated during training: {peak_alloc / 1024**3:.3f} GiB",
        f"  ratio degradation / train step (feed_data in training loop): "
        f"{loop_feed_median:.2f} / {step_median:.2f} ms = "
        f"{loop_feed_median / step_median:.3f}",
        f"  ratio degradation / train step (isolated pass-A feed_data): "
        f"{feed_median:.2f} / {step_median:.2f} ms = {feed_median / step_median:.3f}",
    ]
    return lines


def cpu_list(cpus: set[int]) -> str:
    """Compact CPU set, e.g. {16..31} -> '16-31' (16 CPUs)."""
    ordered = sorted(cpus)
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in [*ordered[1:], None]:
        if cpu is not None and cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(f"{start}-{previous}" if previous != start else f"{start}")
        if cpu is not None:
            start = previous = cpu
    return f"{','.join(ranges)} ({len(ordered)} CPUs)"


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def describe(label: str, values: list[float]) -> str:
    return (
        f"{label:<44} median {pct(values, 50):8.2f}  p90 {pct(values, 90):8.2f}"
        f"  mean {float(np.mean(values)):8.2f}  min {min(values):8.2f}  ms"
    )


def run_text(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"<{' '.join(cmd)} failed: {error}>"
    if result.returncode != 0:
        return f"<{' '.join(cmd)} exit {result.returncode}: {result.stderr.strip()}>"
    return result.stdout.strip()


def stage_table(timer: StageTimer, feed_ms: list[float], iters: int) -> list[str]:
    names = sorted({name for batch in timer.history for name in batch})
    rows = []
    for name in names:
        self_ms = [batch.get(name, [0.0, 0.0, 0])[1] * 1e3 for batch in timer.history]
        incl_ms = [batch.get(name, [0.0, 0.0, 0])[0] * 1e3 for batch in timer.history]
        calls = sum(int(batch.get(name, [0.0, 0.0, 0])[2]) for batch in timer.history)
        rows.append((name, self_ms, incl_ms, calls))
    rows.sort(key=lambda row: sum(row[1]), reverse=True)

    total_feed = float(np.sum(feed_ms))
    lines = [
        f"{'stage':<30} {'self med':>9} {'self p90':>9} {'self mean':>9} {'share%':>7}"
        f" {'incl med':>9} {'calls/b':>7} {'incl/call':>9}  parent / where",
        "-" * 120,
    ]
    stage_self_sum = 0.0
    off_gpu_sum = 0.0
    for name, self_ms, incl_ms, calls in rows:
        share = 100.0 * float(np.sum(self_ms)) / total_feed
        stage_self_sum += float(np.sum(self_ms))
        if name in OFF_GPU_STAGES:
            off_gpu_sum += float(np.sum(self_ms))
        parent = timer.parents.get(name)
        where = STAGE_WHERE.get(name, "GPU")
        note = f"in {parent}; {where}" if parent else where
        lines.append(
            f"{name:<30} {pct(self_ms, 50):9.3f} {pct(self_ms, 90):9.3f}"
            f" {float(np.mean(self_ms)):9.3f} {share:7.1f} {pct(incl_ms, 50):9.3f}"
            f" {calls / iters:7.2f} {float(np.sum(incl_ms)) / calls:9.3f}  {note}"
        )
    unattributed = total_feed - stage_self_sum
    lines.append("-" * 120)
    lines.append(
        f"{'sum of stage self times':<30} {'':>9} {'':>9} {stage_self_sum / iters:9.3f}"
        f" {100.0 * stage_self_sum / total_feed:7.1f}"
    )
    lines.append(
        f"{'unattributed (glue, clamp/round)':<30} {'':>9} {'':>9} {unattributed / iters:9.3f}"
        f" {100.0 * unattributed / total_feed:7.1f}"
    )
    lines.append(
        f"{'off-GPU stages (CPU round trip)':<30} {'':>9} {'':>9} {off_gpu_sum / iters:9.3f}"
        f" {100.0 * off_gpu_sum / total_feed:7.1f}  " + ", ".join(OFF_GPU_STAGES)
    )
    return lines


def census_lines(counts: dict[tuple[str, int], int], iters: int) -> list[str]:
    if not counts:
        return ["no implicit synchronising CUDA operations reported inside feed_data"]
    lines = [f"{'syncs/batch':>11}  location  |  source line", "-" * 120]
    for (filename, lineno), count in sorted(counts.items(), key=lambda kv: -kv[1]):
        source = linecache.getline(filename, lineno).strip()
        location = (
            osp.relpath(filename, ROOT) if filename.startswith(ROOT) else filename
        )
        lines.append(f"{count / iters:11.2f}  {location}:{lineno}  |  {source[:80]}")
    return lines


def config_lines(model: BaseModel, overrides: dict[str, str]) -> list[str]:
    opt = model.opt
    lines = ["Overrides applied in memory to the option file:"]
    lines += [f"  {name}: {value}" for name, value in overrides.items()]
    lines.append("Effective OTF options (model.opt):")
    lines += [
        f"  {name}: {getattr(opt, name)}"
        for name in opt.__struct_fields__
        if name.startswith(OTF_OPTION_PREFIXES)
    ]
    lines.append("Effective train dataset options:")
    train_opt = opt.datasets["train"]
    lines += [
        f"  {name}: {getattr(train_opt, name)}"
        for name in train_opt.__struct_fields__
        if getattr(train_opt, name) is not None
    ]
    return lines


def main() -> None:
    args = parse_args()
    # WSL's nvidia-smi cannot list processes; memory and utilisation taken
    # before this process touches CUDA show whether something else is running.
    load_before = tuple(round(load, 2) for load in os.getloadavg())
    gpu_before = run_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    opt, opt_args = load_options(args)
    overrides = apply_overrides(opt, args)
    assert opt.manual_seed is not None
    assert opt.rank is not None

    # Same process-level settings as train.py.
    if opt.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=opt.deterministic_warn_only)
    else:
        torch.backends.cudnn.benchmark = True
    set_random_seed(opt.manual_seed + opt.rank)
    logger = get_root_logger(logger_name="traiNNer")

    train_loader, train_sampler, _, _, _ = create_train_val_dataloader(
        opt, opt_args, False, logger
    )
    if train_loader is None or train_sampler is None:
        raise RuntimeError("create_train_val_dataloader returned no train loader.")
    if opt.fast_matmul:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.fp32_precision = "tf32"

    model = build_model(opt)
    if type(model) is not RealESRGANModel:
        raise RuntimeError(
            f"build_model returned {type(model).__name__}; this profile needs RealESRGANModel "
            "(high_order_degradation: true, dataroot_lq_prob: 0)."
        )
    if opt.datasets["train"].prefetch_mode != "cuda":
        raise RuntimeError(
            "This profile expects prefetch_mode: cuda, as in the template."
        )
    prefetcher = CUDAPrefetcher(train_loader, opt)
    train_sampler.set_epoch(0)
    prefetcher.reset()

    timer = StageTimer()
    install_stage_timers(timer, model)
    warm_up(model, prefetcher, args.warmup)
    torch.cuda.synchronize()
    baseline_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    fetch_ms, feed_a, feed_b = run_interleaved(model, prefetcher, timer, args.iters)
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    census = (
        sync_census(model, prefetcher, args.sync_census) if args.sync_census > 0 else {}
    )

    train_report: list[str] = []
    if args.train_steps > 0:
        assert model.opt.train is not None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        warmup_steps, train_records = run_train_steps(
            model,
            prefetcher,
            model.opt.train.warmup_iter,
            args.warmup,
            args.train_steps,
        )
        train_report = [
            *training_lines(
                model,
                warmup_steps,
                train_records,
                feed_a,
                torch.cuda.max_memory_allocated(),
            ),
            "",
        ]

    gib = 1024**3
    lq_shape = tuple(model.lq.shape) if model.lq is not None else None
    gt_shape = tuple(model.gt.shape) if model.gt is not None else None
    stage_sum_mean = (
        sum(rec[1] for batch_rec in timer.history for rec in batch_rec.values())
        * 1e3
        / args.iters
    )
    git_head = run_text(["git", "rev-parse", "--short", "HEAD"])
    git_modified = run_text(["git", "status", "--porcelain", "--untracked-files=no"])

    report = [
        "=" * 120,
        f"OTF profile  degradations={args.degradations}  dither={args.dither}  "
        f"codecs={args.codecs}  train_steps={args.train_steps}  "
        f"{datetime.now().isoformat(timespec='seconds')}",
        "=" * 120,
        "Command: " + " ".join([sys.executable, *sys.argv]),
        f"Work tree: {ROOT} @ {git_head}; modified tracked files: "
        + (git_modified.replace("\n", ", ") if git_modified else "none"),
        f"torch {torch.__version__}, {torch.cuda.get_device_name()}, "
        f"{os.cpu_count()} CPUs, process affinity {cpu_list(os.sched_getaffinity(0))}",
        f"Before CUDA init: load average {load_before}; GPU {gpu_before}",
        f"Option file: {args.opt}",
        f"Batch: {args.batch} x GT crop {args.gt_size} (dataset crop {args.gt_size + 32}), "
        f"workers {args.workers}, warm-up {args.warmup}, "
        f"{args.iters} timed batches per pass, passes A and B interleaved",
        f"Final tensors: lq {lq_shape}, gt {gt_shape}",
        "",
        "  " + describe("dataloader fetch (prefetcher.next + sync)", fetch_ms),
        "Pass A - feed_data, synchronize only before/after the call (unsynced total):",
        "  " + describe("feed_data", feed_a),
        "Pass B - feed_data with a CUDA sync before/after every stage (adds sync overhead):",
        "  " + describe("feed_data (instrumented)", feed_b),
        f"  sum of stage self times, mean per batch: {stage_sum_mean:.2f} ms",
        f"  gap pass B - pass A (mean): {float(np.mean(feed_b)) - float(np.mean(feed_a)):.2f} ms"
        " = sync overhead + lost CPU/GPU overlap",
        "",
        "GPU memory (torch allocator, cuda:0):",
        f"  allocated before timed passes (model, EMA, OTF queue, warm-up state): "
        f"{baseline_alloc / gib:.3f} GiB",
        f"  peak allocated during passes A+B: {peak_alloc / gib:.3f} GiB "
        f"(+{(peak_alloc - baseline_alloc) / gib:.3f} GiB over the baseline)",
        f"  peak reserved during passes A+B: {peak_reserved / gib:.3f} GiB",
        "",
        "Per-stage breakdown, pass B (ms per batch; self = exclusive of nested stages;",
        "share% = stage self time / pass-B feed_data time over all timed batches):",
        *stage_table(timer, feed_b, args.iters),
        "",
        f"Implicit host syncs inside feed_data (torch sync debug mode, {args.sync_census} batches):",
        *census_lines(census, max(args.sync_census, 1)),
        "",
        *train_report,
        *config_lines(model, overrides),
        "",
    ]
    text = "\n".join(report)
    print(text)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "a" if args.append else "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
