"""The OTF degradation chain (RealESRGANModel.feed_data, its paired crop and MoA)
draws only from the project RNG and torch's seeded generators, so a seeded run
repeats exactly, whatever is logged in between.

Python's random is not safe for this: rich's log handler draws from it on every
log record (rich.style picks a link id for the record's file-path link), so the
resize, codec, combo, crop and MoA picks used to change with print_freq,
validation and checkpoint messages.

The feed_data tests run on CUDA (NLMeans has no CPU path) with a 4-image batch.
"""

from __future__ import annotations

import ast
import random
from collections.abc import Callable
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import default_collate
from traiNNer.data.realesrgan_dataset import RealESRGANDataset
from traiNNer.data.transforms import paired_random_crop, paired_random_crop_multi_gt
from traiNNer.models.realesrgan_model import (
    RealESRGANModel,
    ThickLines,
    _weighted_choice,
)
from traiNNer.ops.batchaug import BatchAugment
from traiNNer.utils import DiffJPEG, get_root_logger
from traiNNer.utils.misc import set_random_seed
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.rng import RNG
from traiNNer.utils.types import DataFeed

ROOT = Path(__file__).resolve().parents[1]
BATCH = 4
ITERS = 3

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="feed_data's NLMeans stage needs CUDA"
)

# Every draw that moved off Python's random is reachable: the combo pick, both
# resize up/down/keep picks and their modes, the final resize mode, the codec
# pick (JPEG and WebP, both stages), the paired crop and the MoA pick with the
# downup/up sampling picks. The custom degradations all run when their combo
# (or the global list) makes them active.
OPTIONS = {
    "name": "otf_rng_stream",
    "scale": 2,
    "num_gpu": 1,
    "path": {},
    "high_order_degradation": True,
    "queue_size": 2 * BATCH,
    "lq_usm": True,
    "thicklines_prob": 0.5,
    "otf_degradation_combos": [
        ["ntsc", "rainbow", "ghosting"],
        ["lowpass", "overshoot", "shift"],
        ["subsampling", "dithering", "interlace"],
    ],
    "otf_degradation_combo_weights": [1.0, 2.0, 3.0],
    "otf_no_combo_weight": 0.1,
    "otf_global_degradations": ["compress", "hf_noise", "nlmeans", "scanline"],
    "ntsc_prob": 1.0,
    "rainbow_prob": 1.0,
    "ghosting_prob": 1.0,
    "interlace_prob": 1.0,
    "scanline_prob": 1.0,
    "lowpass_prob": 1.0,
    "overshoot_prob": 1.0,
    "shift_prob": 1.0,
    "subsampling_prob": 1.0,
    "dithering_prob": 1.0,
    "dithering_types": ["floydsteinberg", "quantize", "order"],
    "dithering_palette_mode": 0.5,
    "otf_hf_noise_prob": 1.0,
    "otf_hf_noise_denoise_lq": True,
    "compress_prob": 1.0,
    "compress_prob2": 1.0,
    "compress_algorithms": ["jpeg", "webp"],
    "compress_algorithm_probs": [0.5, 0.5],
    "compress_algorithms2": ["jpeg", "webp"],
    "compress_algorithm_probs2": [0.5, 0.5],
    "datasets": {
        "train": {
            "name": "otf_rng_stream",
            "type": "RealESRGANDataset",
            "phase": "train",
            "scale": 2,
            "dataroot_gt": [str(ROOT / "datasets" / "train" / "dataset1" / "hr")],
            "io_backend": {"type": "disk"},
            "gt_size": 64,
            "use_hflip": True,
            "use_rot": True,
        }
    },
    "train": {
        "total_iter": 1,
        "use_moa": True,
        "moa_augs": ["none", "downup", "up", "cutblur"],
        "moa_probs": [0.1, 0.4, 0.3, 0.2],
    },
}


def _options() -> ReduxOptions:
    return msgspec.convert(OPTIONS, ReduxOptions)


def _real_batch(monkeypatch: pytest.MonkeyPatch, opt: ReduxOptions) -> DataFeed:
    """BATCH items of the real RealESRGANDataset (image crops and blur/sinc
    kernels), collated as the train loader collates them."""
    set_random_seed(0)
    monkeypatch.setattr(RNG, "_rng", np.random.default_rng(0))
    dataset = RealESRGANDataset(opt.datasets["train"])
    return default_collate([dataset[index] for index in range(BATCH)])


def _otf_model(opt: ReduxOptions) -> RealESRGANModel:
    """The attributes feed_data reads, set as RealESRGANModel.__init__ and
    BaseModel set them, without building networks, losses or optimizers."""
    assert opt.train is not None
    model = object.__new__(RealESRGANModel)
    model.opt = opt
    model.is_train = True
    model.device = torch.device("cuda")
    model.memory_format = torch.preserve_format
    model.lq = None
    model.gt = None
    model.queue_lr = None
    model.queue_gt = None
    model.queue_ptr = 0
    model.queue_size = opt.queue_size
    model.kernel1 = None
    model.kernel2 = None
    model.sinc_kernel = None
    model.jpeger = DiffJPEG(differentiable=False).to(model.device)
    model.thicklines = ThickLines().to(model.device)
    model.otf_debug = False
    model.otf_debug_limit = 0
    model.batch_augment = BatchAugment(opt.scale, opt.train)
    # Read by feed_data in trees whose RealESRGANModel carries forensic
    # provenance and sample-provenance queues; None means disabled, and the
    # attributes are harmless where the model lacks them.
    for name in (
        "forensic_queue_rng",
        "forensic_degradation_cpu_rng_state",
        "forensic_degradation_cuda_rng_state",
        "queue_sample_indices",
        "queue_sample_paths",
        "batch_sample_indices",
        "batch_sample_paths",
    ):
        setattr(model, name, None)
    return model


def _seeded_run(
    monkeypatch: pytest.MonkeyPatch,
    opt: ReduxOptions,
    batch: DataFeed,
    seed: int,
    before_each: Callable[[], None] | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """ITERS feed_data calls on a fresh model, seeded as train.py seeds a run
    (set_random_seed, then the project RNG from the same seed)."""
    set_random_seed(seed)
    monkeypatch.setattr(RNG, "_rng", np.random.default_rng(seed))
    model = _otf_model(opt)
    results: list[tuple[Tensor, Tensor]] = []
    for _ in range(ITERS):
        if before_each is not None:
            before_each()
        model.feed_data(batch)
        assert model.lq is not None
        assert model.gt is not None
        results.append((model.lq.clone(), model.gt.clone()))
    return results


def _same(
    first: list[tuple[Tensor, Tensor]], second: list[tuple[Tensor, Tensor]]
) -> bool:
    return all(
        lq_a.shape == lq_b.shape
        and torch.equal(lq_a, lq_b)
        and gt_a.shape == gt_b.shape
        and torch.equal(gt_a, gt_b)
        for (lq_a, gt_a), (lq_b, gt_b) in zip(first, second, strict=True)
    )


@requires_cuda
def test_same_seed_repeats_the_otf_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    opt = _options()
    batch = _real_batch(monkeypatch, opt)

    first = _seeded_run(monkeypatch, opt, batch, seed=11)
    again = _seeded_run(monkeypatch, opt, batch, seed=11)
    other = _seeded_run(monkeypatch, opt, batch, seed=12)

    assert _same(first, again)
    assert not _same(first, other)


@requires_cuda
def test_logging_does_not_change_the_otf_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opt = _options()
    batch = _real_batch(monkeypatch, opt)
    logger = get_root_logger()

    def log_line() -> None:
        # A log record through the real rich handler, and a direct draw so the
        # test does not rely on how many draws rich makes per record.
        logger.info("otf rng stream test: a log line between iterations")
        random.random()

    quiet = _seeded_run(monkeypatch, opt, batch, seed=11)
    logged = _seeded_run(monkeypatch, opt, batch, seed=11, before_each=log_line)

    assert _same(quiet, logged)


def _python_random_uses(tree: ast.AST) -> list[str]:
    """`import random`, `from random import ...`, `random.x` and `np.random.x`
    nodes (comments and docstrings are not nodes)."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name == "random"
            ]
        elif isinstance(node, ast.ImportFrom) and node.module == "random":
            found.append(f"line {node.lineno}: from random import ...")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "random" or (
                node.value.id in ("np", "numpy") and node.attr == "random"
            ):
                found.append(f"line {node.lineno}: {node.value.id}.{node.attr}")
    return found


# File -> the functions on the OTF path (None: the whole module).
OTF_PATH_SOURCES: dict[str, tuple[str, ...] | None] = {
    "traiNNer/models/realesrgan_model.py": None,
    "traiNNer/ops/batchaug.py": None,
    "traiNNer/data/transforms.py": (
        "paired_random_crop",
        "paired_random_crop_multi_gt",
        "paired_random_crop_list",
    ),
}


@pytest.mark.parametrize("relative_path", list(OTF_PATH_SOURCES))
def test_otf_path_does_not_draw_from_global_generators(relative_path: str) -> None:
    module = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    functions = OTF_PATH_SOURCES[relative_path]
    if functions is None:
        scopes: list[ast.AST] = [module]
    else:
        scopes = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in functions
        ]
        assert {node.name for node in scopes} == set(functions)

    uses = [use for scope in scopes for use in _python_random_uses(scope)]
    assert uses == []


def test_weighted_choice_keeps_the_random_choices_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """population[i] with probability weights[i] / sum(weights), uniform without
    weights, and the population's own element (a str stays a str)."""
    monkeypatch.setattr(RNG, "_rng", np.random.default_rng(3))
    draws = 20_000
    population = ("keep", "up", "down")

    weighted = [_weighted_choice(population, [0.0, 1.0, 3.0]) for _ in range(draws)]
    assert all(type(value) is str for value in weighted)
    assert weighted.count("keep") == 0
    assert weighted.count("down") / draws == pytest.approx(0.75, abs=0.015)

    uniform = [_weighted_choice(population, None) for _ in range(draws)]
    for value in population:
        assert uniform.count(value) / draws == pytest.approx(1 / 3, abs=0.015)

    with pytest.raises(ValueError):
        _weighted_choice(population, [1.0, 2.0])


def test_paired_crops_cover_the_inclusive_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offsets 0..(lq - patch) inclusive, as random.randint gave, with the GT
    crop at scale times the LQ offset."""
    monkeypatch.setattr(RNG, "_rng", np.random.default_rng(5))
    scale = 2
    lq = torch.arange(6 * 7, dtype=torch.float32).view(1, 1, 6, 7)
    gt = torch.nn.functional.interpolate(lq, scale_factor=scale, mode="nearest")
    tops: set[int] = set()
    lefts: set[int] = set()
    for multi in (False, True) * 200:
        if multi:
            gts, lq_crop = paired_random_crop_multi_gt([gt, gt + 1], lq, 8, scale)
            gt_crop = gts[0]
            assert torch.equal(gts[1], gt_crop + 1)
        else:
            gt_crop, lq_crop = paired_random_crop(gt, lq, 8, scale)
        assert lq_crop.shape[-2:] == (4, 4)
        assert gt_crop.shape[-2:] == (8, 8)
        top, left = divmod(int(lq_crop[0, 0, 0, 0]), 7)
        assert torch.equal(gt_crop[..., ::scale, ::scale], lq_crop)
        tops.add(top)
        lefts.add(left)
    assert tops == {0, 1, 2}
    assert lefts == {0, 1, 2, 3}
