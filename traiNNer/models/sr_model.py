import os
import re
import shutil
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from os import path as osp
from typing import Any

import cv2
import torch
from ema_pytorch import EMA
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F  # noqa: N812
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.losses import build_loss
from traiNNer.metrics import calculate_metric
from traiNNer.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from traiNNer.models.base_model import BaseModel
from traiNNer.training_recovery import (
    MAX_CONSECUTIVE_NONFINITE_ATTEMPTS,
    PersistentNonfiniteError,
)
from traiNNer.utils import get_root_logger, imwrite, tensor2img
from traiNNer.utils.color_util import pixelformat2rgb_pt, rgb2pixelformat_pt
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import loss_type_to_label
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


def _get_total_gradient_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    """Compute the total L2 norm with one reduction for uniform dense grads."""
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("cannot compute a gradient norm without gradients")

    first = gradients[0]
    if all(
        not gradient.is_sparse
        and gradient.device == first.device
        and gradient.dtype == first.dtype
        for gradient in gradients
    ):
        if len(gradients) == 1:
            return torch.linalg.vector_norm(first, 2).detach()
        return torch.linalg.vector_norm(
            torch.cat([gradient.reshape(-1) for gradient in gradients]), 2
        ).detach()

    return get_total_norm(
        gradients,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=None,
    ).detach()


class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

        # use amp
        self.use_amp = self.opt.use_amp
        self.use_channels_last = self.opt.use_channels_last
        self.memory_format = (
            torch.channels_last
            if self.use_amp and self.use_channels_last
            else torch.preserve_format
        )
        self.amp_dtype = torch.bfloat16 if self.opt.amp_bf16 else torch.float16
        self.use_compile = self.opt.use_compile
        self.compile_discriminator = self.opt.compile_discriminator
        self.distillation_opt = (
            opt.train.distillation
            if opt.is_train
            and opt.train is not None
            and opt.train.distillation is not None
            and opt.train.distillation.enabled
            else None
        )
        self.distillation = self.distillation_opt is not None
        self.cudagraph_stable_gradients = False
        self.cudagraph_stable_discriminator_gradients = False
        self._validation_inference_net: nn.Module | None = None
        self._validation_prepared_key: tuple[int, int] | None = None

        # define network
        assert opt.network_g is not None, "network_g must be defined"
        self.net_g = build_network({**opt.network_g, "scale": opt.scale})

        # load pretrained models
        if self.opt.path.pretrain_network_g is not None:
            self.load_network(
                self.net_g,
                self.opt.path.pretrain_network_g,
                self.opt.path.strict_load_g,
                self.opt.path.param_key_g,
            )

        # define teacher network
        if opt.network_g_teacher is not None:
            self.net_g_teacher = build_network(
                {**opt.network_g_teacher, "scale": opt.scale}
            )

            if self.opt.path.pretrain_network_g_teacher is None:
                raise ValueError(
                    "pretrain_network_g_teacher must be defined when using network_g_teacher"
                )
            self.load_network(
                self.net_g_teacher,
                self.opt.path.pretrain_network_g_teacher,
                True,
                None,
            )
            self.net_g_teacher.requires_grad_(False)
        elif self.distillation:
            raise ValueError("distillation.enabled requires network_g_teacher")

        self.net_g = self.model_to_device(self.net_g, compile=self.opt.use_compile)
        if self.net_g_teacher is not None:
            compile_teacher = self.opt.use_compile and (
                not self.distillation
                or (
                    self.distillation_opt is not None
                    and self.distillation_opt.compile_teacher
                )
            )
            self.net_g_teacher = self.model_to_device(
                self.net_g_teacher,
                compile=compile_teacher,
                wrap_distributed=False,
            )
            self.net_g_teacher.eval()

        self.lq: Tensor | None = None
        self.gt: Tensor | None = None
        self.output: Tensor | None = None
        logger = get_root_logger()

        if self.use_amp:
            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning(
                    "bf16 was enabled for AMP but the current GPU does not support bf16. Falling back to float16 for AMP. Disable bf16 to hide this warning (amp_bf16: false)."
                )
                self.amp_dtype = torch.float16

            network_g_name = opt.network_g["type"]
            if (
                self.amp_dtype == torch.float16
                and network_g_name.lower() in ARCHS_WITHOUT_FP16
            ):
                if torch.cuda.is_bf16_supported():
                    logger.warning(
                        "AMP with fp16 was enabled but network_g [bold]%s[/bold] does not support fp16. Falling back to bf16.",
                        network_g_name,
                        extra={"markup": True},
                    )
                    self.amp_dtype = torch.bfloat16
                else:
                    logger.warning(
                        "AMP with fp16 was enabled but network_g [bold]%s[/bold] does not support fp16. Disabling AMP.",
                        network_g_name,
                        extra={"markup": True},
                    )
                    self.use_amp = False
        elif self.amp_dtype == torch.bfloat16:
            logger.warning(
                "bf16 was enabled without AMP and will have no effect. Enable AMP to use bf16 (use_amp: true)."
            )

        if self.use_amp:
            logger.info(
                "Using Automatic Mixed Precision (AMP) with fp32 and %s.",
                "bf16" if self.amp_dtype == torch.bfloat16 else "fp16",
            )

            if self.use_channels_last:
                logger.info("Using channels last memory format.")

        if self.opt.fast_matmul:
            logger.info(
                "Fast matrix multiplication and convolution operations (fast_matmul) enabled, trading precision for performance."
            )

        if self.is_train and self.opt.train:
            # define network net_d if GAN is enabled
            self.has_gan = False
            gan_opt = self.opt.train.gan_opt

            if not gan_opt:
                if self.opt.train.losses:
                    gan_opts = list(
                        filter(
                            lambda x: x["type"].lower() == "ganloss",
                            self.opt.train.losses,
                        )
                    )
                    if gan_opts:
                        gan_opt = gan_opts[0]

            if gan_opt:
                if gan_opt.get("loss_weight", 0) != 0:
                    self.has_gan = True

            self.net_d = None
            if self.has_gan:
                if self.opt.train.optim_d is None:
                    raise ValueError(
                        "GAN loss requires discriminator optimizer (optim_d). Define optim_d or disable GAN loss."
                    )
                if self.opt.network_d is None:
                    raise ValueError(
                        "GAN loss requires discriminator network (network_d). Define network_d or disable GAN loss."
                    )
                else:
                    self.net_d = build_network(self.opt.network_d)
                    # load pretrained models
                    if self.opt.path.pretrain_network_d is not None:
                        self.load_network(
                            self.net_d,
                            self.opt.path.pretrain_network_d,
                            self.opt.path.strict_load_d,
                            self.opt.path.param_key_d,
                        )
                    self.net_d = self.model_to_device(
                        self.net_d, compile=self.compile_discriminator
                    )

            self.losses = {}

            self.ema_decay = 0
            self.net_g_ema: EMA | None = None

            self.optimizer_g: Optimizer | None = None
            self.optimizer_d: Optimizer | None = None
            self.optimizer_d_index: int | None = None

            self.init_training_settings()

    def _discard_accumulated_gradients(self) -> None:
        if self.optimizer_g is not None:
            self._zero_generator_optimizer(self.optimizer_g)
        if self.optimizer_d is not None:
            self._zero_discriminator_optimizer(self.optimizer_d)

    def init_training_settings(self) -> None:
        self.net_g.train()
        if self.net_g_teacher is not None:
            self.net_g_teacher.eval()
        if self.net_d is not None:
            self.net_d.train()

        train_opt = self.opt.train
        assert train_opt is not None

        logger = get_root_logger()

        enable_gradscaler = self.use_amp and not self.opt.amp_bf16

        self.scaler_g = GradScaler(enabled=enable_gradscaler, device="cuda")
        self.scaler_d = GradScaler(enabled=enable_gradscaler, device="cuda")

        if self.distillation:
            assert self.net_g_teacher is not None
            assert self.distillation_opt is not None
            if self.distillation_opt.output_weight < 0:
                raise ValueError("distillation output_weight must be non-negative")
            if self.distillation_opt.feature_weight < 0:
                raise ValueError("distillation feature_weight must be non-negative")
            if (
                self.distillation_opt.output_weight == 0
                and self.distillation_opt.feature_weight == 0
            ):
                raise ValueError(
                    "distillation requires a positive output_weight or feature_weight"
                )
            if self.distillation_opt.feature_pool_size < 1:
                raise ValueError("distillation feature_pool_size must be at least 1")
            if self.distillation_opt.start_iter < 0:
                raise ValueError("distillation start_iter must be non-negative")
            if self.distillation_opt.warmup_iters < 0:
                raise ValueError("distillation warmup_iters must be non-negative")
            if self.distillation_opt.feature_start_iter < 0:
                raise ValueError("distillation feature_start_iter must be non-negative")
            if self.distillation_opt.feature_warmup_iters < 0:
                raise ValueError(
                    "distillation feature_warmup_iters must be non-negative"
                )
            if (self.distillation_opt.decay_start_iter is None) != (
                self.distillation_opt.end_iter is None
            ):
                raise ValueError(
                    "distillation decay_start_iter and end_iter must be set together"
                )
            if (
                self.distillation_opt.decay_start_iter is not None
                and self.distillation_opt.end_iter is not None
                and self.distillation_opt.end_iter
                <= self.distillation_opt.decay_start_iter
            ):
                raise ValueError(
                    "distillation end_iter must be greater than decay_start_iter"
                )

            if self.distillation_opt.feature_weight == 0:
                self.distillation_feature_pairs = ()
                self.distillation_student_feature_count = 0
                self.distillation_teacher_feature_count = 0
                logger.info(
                    "Using frozen-teacher output distillation from %s.",
                    self.opt.path.pretrain_network_g_teacher,
                )
            else:
                self._setup_feature_distillation_contract()

            if any(
                parameter.requires_grad for parameter in self.net_g_teacher.parameters()
            ):
                raise RuntimeError("the distillation teacher must be fully frozen")

        self.accum_iters = self.opt.datasets["train"].accum_iter

        self.adaptive_d = train_opt.adaptive_d
        self.adaptive_d_ema_decay = train_opt.adaptive_d_ema_decay
        self.adaptive_d_threshold = train_opt.adaptive_d_threshold
        self.ema_decay = train_opt.ema_decay

        if self.ema_decay > 0:
            logger.info(
                "Using Exponential Moving Average (EMA) with decay: %s.", self.ema_decay
            )
            assert self.opt.network_g is not None, "network_g must be defined"

            init_net_g_ema = None

            # load pretrained model
            if self.opt.path.pretrain_network_g_ema is not None:
                init_net_g_ema = build_network(
                    {**self.opt.network_g, "scale": self.opt.scale}
                )
                self.load_network(
                    init_net_g_ema,
                    self.opt.path.pretrain_network_g_ema,
                    self.opt.path.strict_load_g,
                    "params_ema",
                )

            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            switch_iter = train_opt.ema_switch_iter
            if switch_iter == 0:
                switch_iter = None
            self.net_g_ema = EMA(
                self.get_bare_model(self.net_g),
                ema_model=init_net_g_ema,
                beta=self.ema_decay,
                allow_different_devices=True,
                update_after_step=train_opt.ema_update_after_step,
                update_every=1,
                power=train_opt.ema_power,
                use_foreach=train_opt.ema_foreach,
                update_model_with_ema_every=switch_iter,
            ).to(device=self.device, memory_format=self.memory_format)  # pyright: ignore[reportCallIssue]

            assert self.net_g_ema is not None
            self.net_g_ema.step = self.net_g_ema.step.to(device=torch.device("cpu"))
            self.net_g_ema.initted = self.net_g_ema.initted.to(device="cpu")
            if self.use_compile:
                self.prepare_network_for_compile(self.net_g)

        self.grad_clip = train_opt.grad_clip
        self.grad_clip_max_norm = train_opt.grad_clip_max_norm
        if self.grad_clip:
            logger.info(
                "Gradient clipping is enabled with max norm=%f.",
                self.grad_clip_max_norm,
            )

        # define losses

        if train_opt.losses is None:
            train_opt.losses = []
            # old loss format
            old_loss_opts = [
                "pixel_opt",
                "mssim_opt",
                "ms_ssim_l1_opt",
                "perceptual_opt",
                "contextual_opt",
                "dists_opt",
                "hr_inversion_opt",
                "dinov2_opt",
                "topiq_opt",
                "pd_opt",
                "fd_opt",
                "ldl_opt",
                "hsluv_opt",
                "gan_opt",
                "color_opt",
                "luma_opt",
                "avg_opt",
                "bicubic_opt",
            ]
            for opt in old_loss_opts:
                loss = getattr(train_opt, opt)
                if loss is not None:
                    train_opt.losses.append(loss)

        for loss in train_opt.losses:
            assert "type" in loss, "all losses must define type"
            assert "loss_weight" in loss, f"{loss['type']} must define loss_weight"
            if float(loss["loss_weight"]) != 0:
                label = loss_type_to_label(loss["type"])
                if label == "l_g_gan":
                    self.has_gan = True
                self.losses[label] = build_loss(loss).to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765

                if self.use_compile:
                    logger.info(
                        "Loss %s will be compiled. The first iteration may take several minutes.",
                        label,
                    )
                    self.losses[label] = torch.compile(self.losses[label])

        assert self.losses, "At least one loss must be defined."

        if not self.has_gan:
            # warn that discriminator network / optimizer won't be used if enabled
            if self.opt.network_d is not None:
                logger.warning(
                    "Discriminator network (network_d) is defined but GAN loss is disabled. Discriminator network will have no effect."
                )

            if train_opt.optim_d is not None:
                logger.warning(
                    "Discriminator optimizer (optim_d) is defined but GAN loss is disabled. Discriminator optimizer will have no effect."
                )

        # setup batch augmentations
        self.setup_batchaug()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self._prepare_cudagraph_gradient_accumulation()
        self.setup_schedulers()

        # logging features
        # self._setup_feature_hooks()

    def _setup_feature_distillation_contract(self) -> None:
        assert self.net_g_teacher is not None
        assert self.distillation_opt is not None
        logger = get_root_logger()
        student = self.get_bare_model(self.net_g)
        teacher = self.get_bare_model(self.net_g_teacher)
        student_contract = getattr(student, "distillation_feature_names", None)
        teacher_contract = getattr(teacher, "distillation_feature_names", None)
        if not callable(student_contract) or not callable(teacher_contract):
            raise TypeError(
                "feature distillation requires student and teacher architectures "
                "with distillation_feature_names() and feature-returning forwards"
            )
        student_names = tuple(student_contract())
        teacher_names = tuple(teacher_contract())
        architecture_mapping = getattr(student, "distillation_feature_pairs", None)
        if callable(architecture_mapping):
            feature_pairs = tuple(architecture_mapping(teacher))
        elif student_names == teacher_names:
            feature_pairs = tuple(
                (index, index, name, name) for index, name in enumerate(student_names)
            )
        else:
            raise ValueError(
                "student/teacher distillation feature contracts differ and "
                "the student provides no semantic mapping"
            )
        if not feature_pairs:
            raise ValueError("feature distillation requires at least one feature pair")
        for student_index, teacher_index, student_name, teacher_name in feature_pairs:
            if not (0 <= student_index < len(student_names)):
                raise ValueError("student distillation feature index is out of range")
            if not (0 <= teacher_index < len(teacher_names)):
                raise ValueError("teacher distillation feature index is out of range")
            if student_names[student_index] != student_name:
                raise ValueError("student distillation feature mapping is inconsistent")
            if teacher_names[teacher_index] != teacher_name:
                raise ValueError("teacher distillation feature mapping is inconsistent")

        self.distillation_feature_pairs = feature_pairs
        self.distillation_student_feature_count = len(student_names)
        self.distillation_teacher_feature_count = len(teacher_names)
        logger.info(
            "Frozen-teacher distillation maps %d sparse student blocks onto %d "
            "teacher blocks (%d supervised pairs, transform=%s).",
            len(student_names),
            len(teacher_names),
            len(feature_pairs),
            self.distillation_opt.feature_transform,
        )

    def _setup_feature_hooks(self) -> None:
        self._feature_entropy = None  # pyright: ignore[reportUninitializedInstanceVariable]

        def entropy_hook(module: nn.Module, input: Tensor, output: Tensor) -> None:
            with torch.no_grad():
                # Handle SPAB output which is a tuple (out, out1, sim_att)
                if isinstance(output, tuple):
                    x = output[0]
                else:
                    x = output

                if x.dim() == 3:  # (B, L, C)
                    x_avg = x.abs().mean(dim=(0, 1))
                else:  # (B, C, H, W)
                    x_avg = x.abs().mean(dim=(0, 2, 3))
                p = torch.softmax(x_avg, dim=0)
                self._feature_entropy = -(p * torch.log(p + 1e-8)).sum().item()

        # Try HAT/DAT style first
        for name, module in self.net_g.named_modules():
            if "layers.5" in name and name.endswith(
                ("blocks.5", "blocks.4", "blocks.3")
            ):
                module.register_forward_hook(entropy_hook)  # pyright: ignore[reportArgumentType]
                print(f"Entropy hook registered on: {name}")
                return

        # Try SPAN style (block_6, block_5, etc.)
        for block_name in ("block_6", "block_5", "block_4"):
            if hasattr(self.net_g, block_name):
                module = getattr(self.net_g, block_name)
                module.register_forward_hook(entropy_hook)
                print(f"Entropy hook registered on: {block_name}")
                return

        # Try RealPLKSR style - feats is a Sequential with PLKBlocks
        if hasattr(self.net_g, "feats") and isinstance(self.net_g.feats, nn.Sequential):
            for i in range(len(self.net_g.feats) - 1, -1, -1):
                if self.net_g.feats[i].__class__.__name__ == "PLKBlock":
                    self.net_g.feats[i].register_forward_hook(entropy_hook)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
                    print(f"Entropy hook registered on: feats[{i}] (PLKBlock)")
                    return

        # Try FDAT style - groups is a Sequential of SimplifiedResidualGroup
        if hasattr(self.net_g, "groups") and isinstance(
            self.net_g.groups, nn.Sequential
        ):
            # Get the last group's last block
            last_group = self.net_g.groups[-1]
            if hasattr(last_group, "blocks") and len(last_group.blocks) > 0:  # pyright: ignore[reportArgumentType]
                last_block = last_group.blocks[-1]  # pyright: ignore[reportIndexIssue]
                last_block.register_forward_hook(entropy_hook)  # pyright: ignore[reportAttributeAccessIssue]
                print(
                    "Entropy hook registered on: groups[-1].blocks[-1] (SimplifiedDATBlock)"
                )
                return

        print("WARNING: Could not find layer to register entropy hook")

    def setup_optimizers(self) -> None:
        train_opt = self.opt.train
        assert train_opt is not None
        # assert train_opt.optim_g is not None
        optim_params = []
        logger = get_root_logger()
        include_patterns = train_opt.optim_include_params_g or []
        exclude_patterns = train_opt.optim_exclude_params_g or []

        try:
            include_regexes = [re.compile(pattern) for pattern in include_patterns]
            exclude_regexes = [re.compile(pattern) for pattern in exclude_patterns]
        except re.error as exc:
            raise ValueError(
                f"Invalid generator optimizer parameter regex: {exc}"
            ) from exc

        def _selected_for_optim(name: str) -> bool:
            if include_regexes and not any(
                regex.search(name) for regex in include_regexes
            ):
                return False
            if exclude_regexes and any(regex.search(name) for regex in exclude_regexes):
                return False
            return True

        if train_opt.optim_g is not None:
            filtered_trainable: list[str] = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if _selected_for_optim(k):
                        optim_params.append(v)
                    else:
                        filtered_trainable.append(k)
                elif "eval_" in k:
                    pass  # intentionally frozen for reparameterization, skip warning
                else:
                    logger.warning("Params %s will not be optimized.", k)

            if (include_patterns or exclude_patterns) and filtered_trainable:
                logger.info(
                    "Filtered %d trainable generator params from optimizer, up to 10 shown:",
                    len(filtered_trainable),
                )
                for name in filtered_trainable[:10]:
                    logger.info("    %s", name)
            if (include_patterns or exclude_patterns) and len(optim_params) == 0:
                raise ValueError(
                    "Generator optimizer parameter filters excluded all trainable parameters."
                )

            self.optimizer_g = self.get_optimizer(optim_params, train_opt.optim_g)
            self.optimizers.append(self.optimizer_g)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in train_opt.optim_g["type"].upper()
            )
        else:
            logger.warning("!!! net_g will not be optimized. !!!")

        # optimizer d
        if self.net_d is not None:
            assert train_opt.optim_d is not None
            self.optimizer_d = self.get_optimizer(
                self.net_d.parameters(), train_opt.optim_d
            )
            self.optimizer_d_index = len(self.optimizers)
            self.optimizers.append(self.optimizer_d)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in train_opt.optim_d["type"].upper()
            )

    def _record_nonfinite_rejection(self, candidate_iter: int) -> bool:
        """Record one rejected effective update and report a stalled loop."""
        self.nan_count += 1
        self._pending_nonfinite_rejection_count += 1
        limit_reached = self.nan_count >= MAX_CONSECUTIVE_NONFINITE_ATTEMPTS
        if self.nan_count % 100 == 0 or limit_reached:
            get_root_logger().warning(
                "NaN/Inf rejected attempts (count: %d), skipping update",
                self._pending_nonfinite_rejection_count,
            )
            self._pending_nonfinite_rejection_count = 0
        return limit_reached

    def _report_pending_nonfinite_rejections(self, candidate_iter: int) -> None:
        """Report all pending rejections on the next 100-iteration boundary."""
        if candidate_iter % 100 != 0 or self._pending_nonfinite_rejection_count == 0:
            return
        get_root_logger().warning(
            "NaN/Inf rejected attempts (count: %d), skipping update",
            self._pending_nonfinite_rejection_count,
        )
        self._pending_nonfinite_rejection_count = 0

    def _prepare_cudagraph_gradient_accumulation(self) -> None:
        """Allocate persistent grad buffers before the first compiled backward."""
        uses_cudagraphs = self.opt.compile_mode in {"max-autotune", "reduce-overhead"}
        self.cudagraph_stable_gradients = (
            self.use_compile and uses_cudagraphs and self.accum_iters > 1
        )
        self.cudagraph_stable_discriminator_gradients = (
            self.compile_discriminator
            and self.optimizer_d is not None
            and uses_cudagraphs
            and self.accum_iters > 1
        )
        if not (
            self.cudagraph_stable_gradients
            or self.cudagraph_stable_discriminator_gradients
        ):
            return

        optimizer_targets = (
            ("generator", self.optimizer_g, self.cudagraph_stable_gradients),
            (
                "discriminator",
                self.optimizer_d,
                self.cudagraph_stable_discriminator_gradients,
            ),
        )
        for name, optimizer, enabled in optimizer_targets:
            if optimizer is None or not enabled:
                continue
            parameter_count = 0
            element_count = 0
            seen_parameters: set[int] = set()
            with torch.no_grad():
                for parameter_group in optimizer.param_groups:
                    for parameter in parameter_group["params"]:
                        parameter_id = id(parameter)
                        if parameter_id in seen_parameters:
                            continue
                        seen_parameters.add(parameter_id)
                        if not parameter.requires_grad:
                            continue
                        if parameter.grad is None:
                            parameter.grad = torch.zeros_like(
                                parameter, memory_format=torch.preserve_format
                            )
                        else:
                            parameter.grad.zero_()
                        parameter_count += 1
                        element_count += parameter.numel()

            get_root_logger().info(
                "Preallocated stable CUDA Graph gradient buffers for %d %s "
                "parameters (%d elements) across %d accumulation steps.",
                parameter_count,
                name,
                element_count,
                self.accum_iters,
            )

    def _zero_generator_optimizer(self, optimizer: Optimizer) -> None:
        """Clear generator grads without releasing CUDA Graph-owned storage."""
        optimizer.zero_grad(set_to_none=not self.cudagraph_stable_gradients)

    def _zero_discriminator_optimizer(self, optimizer: Optimizer) -> None:
        """Clear discriminator grads without releasing CUDA Graph-owned storage."""
        optimizer.zero_grad(
            set_to_none=not self.cudagraph_stable_discriminator_gradients
        )

    def feed_data(self, data: DataFeed) -> None:
        assert "lq" in data
        self.lq = data["lq"].to(
            self.device,
            memory_format=self.memory_format,
            non_blocking=True,
        )
        if self.net_g_teacher is not None and not self.distillation:
            with torch.inference_mode():
                self.gt = self.net_g_teacher(self.lq).to(
                    memory_format=self.memory_format
                )
        elif "gt" in data:
            self.gt = data["gt"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )

        # moa
        if self.is_train and self.batch_augment and self.gt is not None:
            self.gt, self.lq = self.batch_augment(self.gt, self.lq)

    def _forward_generator_for_training(
        self, lq: Tensor, target: Tensor, current_iter: int
    ) -> Tensor:
        """Run the generator training graph.

        Specialized models can override this without duplicating the optimizer,
        AMP, EMA, discriminator, or accumulation machinery below.
        """
        del target, current_iter
        output = self.net_g(lq)
        assert isinstance(output, Tensor)
        return output

    def _generator_auxiliary_losses(self) -> dict[str, Tensor]:
        """Return already-weighted generator-only auxiliary losses."""
        return {}

    @staticmethod
    def _distillation_distance(
        prediction: Tensor,
        target: Tensor,
        loss_type: str,
        epsilon: float,
    ) -> Tensor:
        difference = prediction.float() - target.float()
        if loss_type == "charbonnier":
            return torch.sqrt(difference.square() + epsilon * epsilon).mean()
        if loss_type == "l1":
            return difference.abs().mean()
        if loss_type == "mse":
            return difference.square().mean()
        raise ValueError(f"unsupported distillation loss: {loss_type}")

    @staticmethod
    def _forward_with_distillation_features(
        network: nn.Module, lq: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        result = network(lq, return_distillation_features=True)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], Tensor)
            or not isinstance(result[1], tuple)
            or not all(isinstance(feature, Tensor) for feature in result[1])
        ):
            raise TypeError(
                "feature distillation networks must return "
                "(output, tuple[feature, ...])"
            )
        return result

    def _distillation_scale(
        self, current_iter: int, start_iter: int, warmup_iters: int
    ) -> float:
        assert self.distillation_opt is not None
        if current_iter < start_iter:
            return 0.0
        if warmup_iters <= 0:
            scale = 1.0
        else:
            elapsed = current_iter - start_iter + 1
            scale = min(1.0, elapsed / warmup_iters)

        decay_start = self.distillation_opt.decay_start_iter
        end_iter = self.distillation_opt.end_iter
        if decay_start is not None and end_iter is not None:
            if current_iter >= end_iter:
                return 0.0
            if current_iter >= decay_start:
                scale *= (end_iter - current_iter) / (end_iter - decay_start)
        return scale

    def _distillation_feature_representation(self, feature: Tensor) -> Tensor:
        assert self.distillation_opt is not None
        feature = feature.float()
        transform = self.distillation_opt.feature_transform
        if transform == "none":
            return feature
        if feature.ndim != 4:
            raise ValueError(
                "distillation features must use BCHW layout, got "
                f"shape={tuple(feature.shape)}"
            )
        if transform == "channel_l2":
            return F.normalize(
                feature,
                p=2,
                dim=1,
                eps=self.distillation_opt.epsilon,
            )
        if transform == "spatial_affinity":
            pool_size = self.distillation_opt.feature_pool_size
            if pool_size > 1:
                feature = F.avg_pool2d(
                    feature,
                    kernel_size=pool_size,
                    stride=pool_size,
                    ceil_mode=True,
                )
            feature = feature.flatten(2)
            feature = F.normalize(
                feature,
                p=2,
                dim=1,
                eps=self.distillation_opt.epsilon,
            )
            return torch.bmm(feature.transpose(1, 2), feature)
        raise ValueError(f"unsupported distillation feature transform: {transform}")

    def _feature_distillation_loss(
        self,
        student_features: tuple[Tensor, ...],
        teacher_features: tuple[Tensor, ...],
    ) -> Tensor:
        assert self.distillation_opt is not None
        if len(student_features) != self.distillation_student_feature_count:
            raise ValueError(
                "runtime student feature count does not match the architecture contract"
            )
        if len(teacher_features) != len(self.distillation_feature_pairs):
            raise ValueError(
                "runtime mapped teacher feature count does not match the architecture contract"
            )

        first_student_index = self.distillation_feature_pairs[0][0]
        total = torch.zeros(
            (), device=student_features[first_student_index].device, dtype=torch.float32
        )
        for (
            student_index,
            _teacher_index,
            student_name,
            teacher_name,
        ), teacher_feature in zip(
            self.distillation_feature_pairs,
            teacher_features,
            strict=True,
        ):
            student_feature = student_features[student_index]
            if student_feature.shape[0] != teacher_feature.shape[0] or (
                student_feature.shape[-2:] != teacher_feature.shape[-2:]
            ):
                raise ValueError(
                    "distillation feature batch/spatial shape mismatch for "
                    f"{student_name!r} <- {teacher_name!r}: "
                    f"student={tuple(student_feature.shape)}, "
                    f"teacher={tuple(teacher_feature.shape)}"
                )
            student_representation = self._distillation_feature_representation(
                student_feature
            )
            teacher_representation = self._distillation_feature_representation(
                teacher_feature
            )
            if student_representation.shape != teacher_representation.shape:
                raise ValueError(
                    "distillation feature representation shape mismatch for "
                    f"{student_name!r} <- {teacher_name!r}: "
                    f"student={tuple(student_representation.shape)}, "
                    f"teacher={tuple(teacher_representation.shape)}"
                )
            total = total + self._distillation_distance(
                student_representation,
                teacher_representation,
                self.distillation_opt.feature_loss,
                self.distillation_opt.epsilon,
            )
        return total / len(self.distillation_feature_pairs)

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        self.begin_training_attempt()
        if (
            self.use_compile or (self.compile_discriminator and self.net_d is not None)
        ) and self.opt.compile_mode in {
            "max-autotune",
            "reduce-overhead",
        }:
            # The trainer intentionally retains outputs and logging tensors
            # across calls. Tell CUDA Graph Trees where the new iteration
            # begins so prior-generation buffers can be reclaimed safely.
            torch.compiler.cudagraph_mark_step_begin()

        # https://github.com/Corpsecreate/neosr/blob/2ee3e7fe5ce485e070744158d4e31b8419103db0/neosr/models/default.py#L328
        # assert self.optimizer_g is not None
        assert self.lq is not None
        assert self.gt is not None
        assert self.scaler_d is not None
        assert self.scaler_g is not None

        candidate_iter = current_iter if apply_gradient else current_iter + 1
        skip_d_update = False

        # optimize net_d
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        n_samples = self.gt.shape[0]
        loss_dict: dict[str, Tensor | float] = OrderedDict()

        lq = rgb2pixelformat_pt(
            self.lq, self.opt.input_pixel_format
        )  # lq: input_pixel_format

        teacher_output_target: Tensor | None = None
        teacher_feature_targets: tuple[Tensor, ...] | None = None
        output_distillation_scale = 0.0
        feature_distillation_scale = 0.0
        if self.distillation:
            assert self.distillation_opt is not None
            output_distillation_scale = self._distillation_scale(
                current_iter,
                self.distillation_opt.start_iter,
                self.distillation_opt.warmup_iters,
            )
            feature_distillation_scale = self._distillation_scale(
                current_iter,
                self.distillation_opt.feature_start_iter,
                self.distillation_opt.feature_warmup_iters,
            )
        output_distillation_active = (
            self.distillation
            and self.distillation_opt is not None
            and self.distillation_opt.output_weight > 0
            and output_distillation_scale > 0
        )
        feature_distillation_active = (
            self.distillation
            and self.distillation_opt is not None
            and self.distillation_opt.feature_weight > 0
            and feature_distillation_scale > 0
        )

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if output_distillation_active or feature_distillation_active:
                assert self.net_g_teacher is not None
                with torch.no_grad():
                    if feature_distillation_active:
                        teacher_raw, teacher_features = (
                            self._forward_with_distillation_features(
                                self.net_g_teacher, lq
                            )
                        )
                        if (
                            len(teacher_features)
                            != self.distillation_teacher_feature_count
                        ):
                            raise ValueError(
                                "runtime teacher feature count does not match "
                                "the architecture contract"
                            )
                        teacher_feature_targets = tuple(
                            teacher_features[teacher_index].detach()
                            for (
                                _student_index,
                                teacher_index,
                                _student_name,
                                _teacher_name,
                            ) in self.distillation_feature_pairs
                        )
                    else:
                        teacher_raw = self.net_g_teacher(lq)
                        assert isinstance(teacher_raw, Tensor)
                    if output_distillation_active:
                        teacher_output_target = pixelformat2rgb_pt(
                            teacher_raw, self.gt, self.opt.output_pixel_format
                        ).detach()

            if self.optimizer_g is not None:
                student_features: tuple[Tensor, ...] | None = None
                if feature_distillation_active:
                    output, student_features = self._forward_with_distillation_features(
                        self.net_g, lq
                    )
                else:
                    output = self._forward_generator_for_training(
                        lq, self.gt, current_iter
                    )  # output: output_pixel_format
                self.output = pixelformat2rgb_pt(
                    output, self.gt, self.opt.output_pixel_format
                )  # self.output: rgb

                assert isinstance(self.output, Tensor)
                l_g_total = self.output.sum() * 0.0

                lq_target = None

                # Log entropy if hook captured it
                if (
                    hasattr(self, "_feature_entropy")
                    and self._feature_entropy is not None
                ):
                    loss_dict["entropy"] = torch.tensor(
                        self._feature_entropy, device=self.device
                    )

                for label, loss in self.losses.items():
                    target = self.gt

                    if loss.loss_weight < 0:
                        if lq_target is None:
                            with torch.inference_mode():
                                lq_target = torch.clamp(
                                    F.interpolate(
                                        self.lq,
                                        scale_factor=self.opt.scale,
                                        mode="bicubic",
                                        antialias=True,
                                    ),
                                    0,
                                    1,
                                )
                        target = lq_target

                    if label == "l_g_gan":
                        assert self.net_d is not None
                        fake_g_pred = self.net_d(self.output)
                        l_g_loss = loss(fake_g_pred, True, is_disc=False)

                        if self.adaptive_d:
                            l_g_gan_ema = (
                                self.adaptive_d_ema_decay * self.l_g_gan_ema
                                + (1 - self.adaptive_d_ema_decay) * l_g_loss.detach()
                            )

                            if (
                                l_g_gan_ema
                                > self.l_g_gan_ema * self.adaptive_d_threshold
                            ):
                                skip_d_update = True
                                assert self.optimizer_d_index is not None
                                self.optimizers_skipped[self.optimizer_d_index] = True
                                # print(current_iter, "skip_d_update")

                            self.l_g_gan_ema = l_g_gan_ema

                    elif label == "l_g_ldl":
                        assert self.net_g_ema is not None, (
                            "ema_decay must be enabled for LDL loss"
                        )
                        with torch.inference_mode():
                            output_ema = pixelformat2rgb_pt(
                                self.net_g_ema(lq),
                                self.gt,
                                self.opt.output_pixel_format,
                            )
                        l_g_loss = loss(self.output, output_ema, target)
                    else:
                        l_g_loss = loss(self.output, target)

                    if isinstance(l_g_loss, dict):
                        for sublabel, loss_val in l_g_loss.items():
                            if loss_val > 0:
                                weighted_loss_val = loss_val * abs(loss.loss_weight)
                                l_g_total += weighted_loss_val
                                loss_dict[f"{label}_{sublabel}"] = weighted_loss_val
                    else:
                        weighted_l_g_loss = l_g_loss * abs(loss.loss_weight)
                        l_g_total += weighted_l_g_loss
                        loss_dict[label] = weighted_l_g_loss

                if output_distillation_active:
                    assert self.distillation_opt is not None
                    assert teacher_output_target is not None
                    output_distillation = (
                        self._distillation_distance(
                            self.output,
                            teacher_output_target,
                            self.distillation_opt.output_loss,
                            self.distillation_opt.epsilon,
                        )
                        * self.distillation_opt.output_weight
                        * output_distillation_scale
                    )
                    l_g_total += output_distillation
                    loss_dict["l_g_distill_output"] = output_distillation
                    loss_dict["distill_output_scale"] = output_distillation_scale

                if feature_distillation_active:
                    assert self.distillation_opt is not None
                    assert teacher_feature_targets is not None
                    assert student_features is not None
                    feature_distillation = (
                        self._feature_distillation_loss(
                            student_features, teacher_feature_targets
                        )
                        * self.distillation_opt.feature_weight
                        * feature_distillation_scale
                    )
                    l_g_total += feature_distillation
                    loss_dict["l_g_distill_features"] = feature_distillation
                    loss_dict["distill_feature_scale"] = feature_distillation_scale

                for label, auxiliary_loss in self._generator_auxiliary_losses().items():
                    if label in loss_dict:
                        raise RuntimeError(
                            f"duplicate generator auxiliary loss label: {label}"
                        )
                    l_g_total += auxiliary_loss
                    loss_dict[label] = auxiliary_loss

                backward_g_loss = l_g_total / self.accum_iters

                loss_is_finite_value = bool(backward_g_loss.isfinite().item())

                if not loss_is_finite_value:
                    limit_reached = self._record_nonfinite_rejection(candidate_iter)
                    self.reject_training_step("nonfinite_generator_loss")
                    for index in range(len(self.optimizers_skipped)):
                        self.optimizers_skipped[index] = True
                    self._discard_accumulated_gradients()
                    if limit_reached:
                        raise PersistentNonfiniteError(
                            "nonfinite_generator_loss", candidate_iter, self.nan_count
                        )
                    return

                # add total generator loss for tensorboard tracking
                loss_dict["l_g_total"] = backward_g_loss

                self.scaler_g.scale(backward_g_loss).backward()

                if apply_gradient:
                    self.scaler_g.unscale_(self.optimizer_g)
                    grad_norm_g = _get_total_gradient_norm(self.net_g.parameters())

                    loss_dict["grad_norm_g"] = grad_norm_g

                    if not bool(torch.isfinite(grad_norm_g).item()):
                        limit_reached = self._record_nonfinite_rejection(candidate_iter)
                        self.reject_training_step("nonfinite_generator_gradient")
                        for index in range(len(self.optimizers_skipped)):
                            self.optimizers_skipped[index] = True
                        self._discard_accumulated_gradients()
                        if self.scaler_g.is_enabled():
                            # unscale_ recorded overflow. Consume it and reset the
                            # per-optimizer phase, retaining GradScaler's backoff.
                            # Do not restore the old scale or step the optimizer.
                            self.scaler_g.update()
                        if limit_reached:
                            raise PersistentNonfiniteError(
                                "nonfinite_generator_gradient",
                                candidate_iter,
                                self.nan_count,
                            )
                        return

                    if self.grad_clip:
                        clip_grads_with_norm_(
                            self.net_g.parameters(),
                            self.grad_clip_max_norm,
                            grad_norm_g,
                            foreach=None,
                        )

                    scale_before = self.scaler_g.get_scale()
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                    scale_after = self.scaler_g.get_scale()
                    loss_dict["scale_g"] = scale_after
                    self.optimizers_skipped[0] = scale_after < scale_before
                    self._zero_generator_optimizer(self.optimizer_g)
            else:
                with torch.inference_mode():
                    self.output = self.net_g(self.lq)
                    assert isinstance(self.output, Tensor)

        cri_gan = self.losses.get("l_g_gan")

        discriminator_configured = (
            self.net_d is not None
            and cri_gan is not None
            and self.optimizer_d is not None
        )
        if discriminator_configured:
            assert self.net_d is not None
            assert self.optimizer_d is not None
            assert cri_gan is not None

            if not skip_d_update:
                for p in self.net_d.parameters():
                    p.requires_grad = True

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    # real
                    real_d_pred = self.net_d(self.gt)
                    l_d_real = cri_gan(real_d_pred, True, is_disc=True)
                    loss_dict["l_d_real"] = l_d_real
                    loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
                    # fake
                    assert self.output is not None
                    fake_d_pred = self.net_d(self.output.detach())
                    l_d_fake = cri_gan(fake_d_pred, False, is_disc=True)
                    loss_dict["l_d_fake"] = l_d_fake
                    loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())

                self.scaler_d.scale((l_d_real + l_d_fake) / self.accum_iters).backward()

                if apply_gradient:
                    self.scaler_d.unscale_(self.optimizer_d)
                    grad_norm_d = _get_total_gradient_norm(self.net_d.parameters())

                    loss_dict["grad_norm_d"] = grad_norm_d

                    if self.grad_clip:
                        clip_grads_with_norm_(
                            self.net_d.parameters(),
                            self.grad_clip_max_norm,
                            grad_norm_d,
                            foreach=None,
                        )
                    scale_before = self.scaler_d.get_scale()
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                    scale_after = self.scaler_d.get_scale()
                    loss_dict["scale_d"] = scale_after
                    assert self.optimizer_d_index is not None
                    self.optimizers_skipped[self.optimizer_d_index] = (
                        scale_after < scale_before
                    )
                    self._zero_discriminator_optimizer(self.optimizer_d)
            elif apply_gradient:
                # A final adaptive-D skip must not leak partial gradients into
                # the next effective batch.
                self._zero_discriminator_optimizer(self.optimizer_d)

        self.loss_samples += n_samples
        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()  # pyright: ignore[reportAttributeAccessIssue]
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples

        self.log_dict = self.reduce_loss_dict(self.log_dict)

        if apply_gradient and not any(self.optimizers_skipped):
            self.nan_count = 0
            self._report_pending_nonfinite_rejections(candidate_iter)

        if self.net_g_ema is not None and apply_gradient:
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.net_g_ema.update()
                # EMA updates parameters in-place while its model stays in eval
                # mode. Force one fresh inference preparation before the next
                # validation sweep, then reuse it for every dataset and image.
                self._validation_prepared_key = None

    def infer_tiled(self, net: nn.Module, lq: torch.Tensor) -> torch.Tensor:
        assert self.opt.val is not None
        tile_size = self.opt.val.tile_size
        tile_overlap = self.opt.val.tile_overlap
        scale = self.opt.scale

        b, c, h, w = lq.shape
        assert b == 1, "Only batch size 1 is supported for tiled inference"

        if h <= tile_size and w <= tile_size:
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                return net(lq)

        pad_h = (tile_size - (h % tile_size)) % tile_size if h > tile_size else 0
        pad_w = (tile_size - (w % tile_size)) % tile_size if w > tile_size else 0

        lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, h_pad, w_pad = lq.shape

        output = torch.zeros((1, c, h_pad * scale, w_pad * scale), device=lq.device)
        weight_map = torch.zeros_like(output)

        hr_tile = tile_size * scale
        wy = torch.linspace(0, 1, hr_tile, device=lq.device)
        wx = torch.linspace(0, 1, hr_tile, device=lq.device)
        wy = 1 - torch.abs(wy - 0.5) * 2
        wx = 1 - torch.abs(wx - 0.5) * 2
        weight = torch.ger(wy, wx).unsqueeze(0).unsqueeze(0)

        stride = tile_size - tile_overlap
        tiles_y = max(1, (h_pad - tile_overlap + stride - 1) // stride)
        tiles_x = max(1, (w_pad - tile_overlap + stride - 1) // stride)

        for y in range(tiles_y):
            for x in range(tiles_x):
                in_y0 = y * stride
                in_x0 = x * stride
                in_y1 = min(in_y0 + tile_size, h_pad)
                in_x1 = min(in_x0 + tile_size, w_pad)

                lq_patch = lq[:, :, in_y0:in_y1, in_x0:in_x1]

                ph, pw = lq_patch.shape[-2:]
                pad_bottom = max(tile_size - ph, 0) if ph < tile_size else 0
                pad_right = max(tile_size - pw, 0) if pw < tile_size else 0

                if pad_bottom > 0 or pad_right > 0:
                    pad_bottom = min(pad_bottom, ph - 1)
                    pad_right = min(pad_right, pw - 1)
                    lq_patch = torch.nn.functional.pad(
                        lq_patch,
                        (0, pad_right, 0, pad_bottom),
                        mode="reflect",
                    )

                out_patch = net(lq_patch)
                out_patch = out_patch[:, :, : ph * scale, : pw * scale]
                w_patch = weight[:, :, : ph * scale, : pw * scale]

                out_y0 = in_y0 * scale
                out_x0 = in_x0 * scale
                out_y1 = out_y0 + ph * scale
                out_x1 = out_x0 + pw * scale

                output[:, :, out_y0:out_y1, out_x0:out_x1] += out_patch * w_patch
                weight_map[:, :, out_y0:out_y1, out_x0:out_x1] += w_patch

        out_final = output / weight_map.clamp(min=1e-6)
        return out_final[:, :, : h * scale, : w * scale]

    def _select_inference_net(self, *, unwrap_compiled: bool) -> nn.Module:
        if self.net_g_ema is not None:
            assert isinstance(self.net_g_ema.ema_model, nn.Module)
            return self.net_g_ema.ema_model

        assert isinstance(self.net_g, nn.Module)
        if unwrap_compiled:
            return self.get_bare_model(self.net_g)
        return self.net_g

    @contextmanager
    def _validation_inference_context(self, current_iter: int) -> Iterator[None]:
        """Prepare inference once and keep it stable for a validation dataset.

        Some architectures build reversible eval-only caches from their training
        parameters. Re-entering ``eval()`` for every image rebuilds those caches;
        doing it once per weight identity keeps validation exact and removes that
        repeated work.
        """
        if self._validation_inference_net is not None:
            raise RuntimeError("validation inference contexts cannot be nested")

        previous_is_train = self.is_train
        self.is_train = False
        schedule_free = bool(
            self.optimizers_schedule_free and self.optimizers_schedule_free[0]
        )
        inference_net = self._select_inference_net(
            unwrap_compiled=previous_is_train and self.use_compile
        )
        was_training = inference_net.training
        prepared_key = (id(inference_net), current_iter)

        try:
            if schedule_free:
                assert self.optimizer_g is not None
                self.optimizer_g.eval()  # pyright: ignore[reportAttributeAccessIssue]

            # A new iteration may have updated EMA parameters in-place while the
            # module remained in eval mode. Its eval transition refreshes those
            # caches exactly once; subsequent datasets at this iteration reuse it.
            if inference_net.training or self._validation_prepared_key != prepared_key:
                inference_net.eval()
            self._validation_prepared_key = prepared_key
            self._validation_inference_net = inference_net
            yield
        finally:
            self._validation_inference_net = None
            if self.net_g_ema is None and self.opt.is_train and was_training:
                inference_net.train()
                self._validation_prepared_key = None
            if schedule_free:
                assert self.optimizer_g is not None
                self.optimizer_g.train()  # pyright: ignore[reportAttributeAccessIssue]
            self.is_train = previous_is_train

    def test(self) -> None:
        inference_net = self._validation_inference_net
        manage_inference_state = inference_net is None
        schedule_free = False
        was_training = False

        if inference_net is None:
            inference_net = self._select_inference_net(
                unwrap_compiled=self.is_train and self.use_compile
            )
            was_training = inference_net.training
            schedule_free = bool(
                self.optimizers_schedule_free and self.optimizers_schedule_free[0]
            )
            if schedule_free:
                assert self.optimizer_g is not None
                self.optimizer_g.eval()  # pyright: ignore[reportAttributeAccessIssue]
            inference_net.eval()

        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                assert self.lq is not None
                lq = rgb2pixelformat_pt(
                    self.lq, self.opt.input_pixel_format
                )  # lq: input_pixel_format

                assert self.opt.val is not None
                with torch.inference_mode():
                    if self.opt.val.tile_size > 0:
                        tmp_out = self.infer_tiled(inference_net, lq)
                    else:
                        tmp_out = inference_net(lq)
                    self.output = pixelformat2rgb_pt(
                        tmp_out, self.gt, self.opt.output_pixel_format
                    )
        finally:
            if manage_inference_state:
                if self.net_g_ema is None and self.opt.is_train and was_training:
                    inference_net.train()
                    self._validation_prepared_key = None
                if schedule_free:
                    assert self.optimizer_g is not None
                    self.optimizer_g.train()  # pyright: ignore[reportAttributeAccessIssue]

    def dist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        if self.opt.rank == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )

    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        with self._validation_inference_context(current_iter):
            self._nondist_validation(
                dataloader,
                current_iter,
                tb_logger,
                save_img,
                multi_val_datasets,
            )

    def _nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        assert isinstance(dataloader.dataset, BaseDataset)
        assert self.opt.val is not None
        assert self.opt.path.visualization is not None

        dataset_name = dataloader.dataset.opt.name

        if self.with_metrics:
            assert self.opt.val.metrics is not None
            if len(self.metric_results) == 0:  # only execute in the first run
                self.metric_results: dict[str, Any] = dict.fromkeys(
                    self.opt.val.metrics.keys(), 0
                )
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if self.with_metrics:
            self.metric_results = dict.fromkeys(self.metric_results, 0)

        metric_data = {}
        pbar = None
        if self.use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        logger = get_root_logger()
        if save_img and len(dataloader) > 0:
            logger.info(
                "Saving %d validation images to %s.",
                len(dataloader),
                clickable_file_path(
                    self.opt.path.visualization, "visualization folder"
                ),
            )

        gt_key = "img2"
        run_metrics = self.with_metrics
        metrics_cfg = self.opt.val.metrics if run_metrics else None
        save_executor: ThreadPoolExecutor | None = None
        save_futures = []
        if save_img:
            # Overlap PNG encoding/writes with next validation iterations.
            max_save_workers = max(1, min(8, (os.cpu_count() or 1) // 2))
            save_executor = ThreadPoolExecutor(
                max_workers=max_save_workers, thread_name_prefix="valsave"
            )

        # Fast path for built-in PSNR/SSIM: compute directly on tensors to avoid
        # expensive tensor->numpy conversions per validation image.
        tensor_metric_cfg: dict[str, tuple[str, int, bool]] = {}
        fallback_metric_cfg: dict[str, dict[str, Any]] = {}
        pending_tensor_metrics: list[tuple[str, Tensor]] = []
        if run_metrics and metrics_cfg is not None:
            for metric_name, metric_opt in metrics_cfg.items():
                metric_type = metric_opt.get("type")
                if metric_type == "calculate_psnr":
                    tensor_metric_cfg[metric_name] = (
                        "psnr",
                        int(metric_opt.get("crop_border", 0)),
                        bool(metric_opt.get("test_y_channel", False)),
                    )
                elif metric_type == "calculate_ssim":
                    tensor_metric_cfg[metric_name] = (
                        "ssim",
                        int(metric_opt.get("crop_border", 0)),
                        bool(metric_opt.get("test_y_channel", False)),
                    )
                else:
                    fallback_metric_cfg[metric_name] = metric_opt

            if tensor_metric_cfg:
                logger.info(
                    "Validation metric fast path enabled for %s.",
                    ", ".join(tensor_metric_cfg.keys()),
                )

        for val_data in dataloader:
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            assert self.output is not None
            output_tensor = self.output
            gt_tensor = self.gt
            sr_img = None

            # Fast tensor metrics (PSNR/SSIM).
            if run_metrics and gt_tensor is not None and tensor_metric_cfg:
                # Match legacy metric semantics used by tensor2img() + numpy metrics:
                # clamp to [0, 1], then quantize to uint8 grid.
                output_metric_tensor = (
                    torch.round(output_tensor.detach().float().clamp(0, 1) * 255.0)
                    / 255.0
                )
                gt_metric_tensor = (
                    torch.round(gt_tensor.detach().float().clamp(0, 1) * 255.0) / 255.0
                )
                for metric_name, (
                    metric_kind,
                    crop_border,
                    test_y_channel,
                ) in tensor_metric_cfg.items():
                    if metric_kind == "psnr":
                        metric_val = calculate_psnr_pt(
                            output_metric_tensor,
                            gt_metric_tensor,
                            crop_border=crop_border,
                            test_y_channel=test_y_channel,
                        )
                    else:
                        metric_val = calculate_ssim_pt(
                            output_metric_tensor,
                            gt_metric_tensor,
                            crop_border=crop_border,
                            test_y_channel=test_y_channel,
                        )
                    # Keep scalar results on-device until the dataset is done.
                    # Calling item() here serializes every image with the GPU.
                    pending_tensor_metrics.append(
                        (metric_name, metric_val.mean().detach())
                    )

            # Fallback metrics keep existing numpy path.
            if run_metrics and fallback_metric_cfg:
                metric_data.clear()
                sr_img = tensor2img(output_tensor, to_bgr=False)
                metric_data["img"] = sr_img
                if gt_tensor is not None:
                    metric_data[gt_key] = tensor2img(gt_tensor, to_bgr=False)
                for name, opt_ in fallback_metric_cfg.items():
                    result = calculate_metric(metric_data, opt_, self.device)
                    self.metric_results[name] += result

            # Release per-image references; keep the CUDA allocator cache hot so
            # subsequent images can reuse those allocations without a device sync.
            self.lq = None
            self.gt = None
            self.output = None

            save_img_dir = None

            if save_img:
                if sr_img is None:
                    sr_img = tensor2img(output_tensor, to_bgr=False)
                if self.opt.is_train:
                    if multi_val_datasets:
                        save_img_dir = osp.join(
                            self.opt.path.visualization, f"{dataset_name} - {img_name}"
                        )
                    else:
                        assert dataloader.dataset.opt.dataroot_lq is not None, (
                            "dataroot_lq is required for val set"
                        )
                        lq_path = val_data["lq_path"][0]

                        # multiple root paths are supported, find the correct root path for each lq_path
                        normalized_lq_path = osp.normpath(lq_path)

                        matching_root = None
                        for root in dataloader.dataset.opt.dataroot_lq:
                            normalized_root = osp.normpath(root)
                            if normalized_lq_path.startswith(normalized_root + osp.sep):
                                matching_root = root
                                break

                        if matching_root is None:
                            raise ValueError(
                                f"The lq_path {lq_path} does not match any of the provided dataroot_lq paths."
                            )

                        save_img_dir = osp.join(
                            self.opt.path.visualization,
                            osp.relpath(
                                osp.splitext(lq_path)[0],
                                matching_root,
                            ),
                        )
                    save_img_path = osp.join(
                        save_img_dir, f"{img_name}_{current_iter:06d}.png"
                    )
                elif self.opt.val.suffix:
                    save_img_path = osp.join(
                        self.opt.path.visualization,
                        dataset_name,
                        f"{img_name}_{self.opt.val.suffix}.png",
                    )
                else:
                    save_img_path = osp.join(
                        self.opt.path.visualization,
                        dataset_name,
                        f"{img_name}.png",
                    )
                if save_img_dir is not None:
                    os.makedirs(save_img_dir, exist_ok=True)
                sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                if save_executor is not None:
                    save_futures.append(
                        save_executor.submit(imwrite, sr_img_bgr, save_img_path)
                    )
                else:
                    imwrite(sr_img_bgr, save_img_path)
                if (
                    self.opt.is_train
                    and not self.first_val_completed
                    and "lq_path" in val_data
                ):
                    assert save_img_dir is not None
                    lr_img_target_path = osp.join(save_img_dir, f"{img_name}_lr.png")
                    if not os.path.exists(lr_img_target_path):
                        if save_executor is not None:
                            save_futures.append(
                                save_executor.submit(
                                    shutil.copy,
                                    val_data["lq_path"][0],
                                    lr_img_target_path,
                                )
                            )
                        else:
                            shutil.copy(val_data["lq_path"][0], lr_img_target_path)

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if pbar is not None:
            pbar.close()

        if pending_tensor_metrics:
            # Preserve the original per-image floating-point values and Python
            # summation order while paying for one GPU-to-CPU synchronization.
            host_metric_values = (
                torch.stack([value for _, value in pending_tensor_metrics])
                .cpu()
                .tolist()
            )
            for (metric_name, _), metric_value in zip(
                pending_tensor_metrics, host_metric_values, strict=True
            ):
                self.metric_results[metric_name] += float(metric_value)

        if save_executor is not None:
            for future in save_futures:
                # Propagate write/copy errors before finishing validation.
                future.result()
            save_executor.shutdown(wait=True)

        if run_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= len(dataloader)
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.first_val_completed = True

    def _log_validation_metric_values(
        self, current_iter: int, dataset_name: str, tb_logger: SummaryWriter | None
    ) -> None:
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric:<5}: {value:7.4f}"
            if len(self.best_metric_results) > 0:
                log_str += (
                    f"\tBest: {self.best_metric_results[dataset_name][metric]['val']:7.4f} @ "
                    f"{self.best_metric_results[dataset_name][metric]['iter']:9,} iter"
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def get_current_visuals(self) -> dict[str, Tensor]:
        assert self.output is not None
        assert self.lq is not None

        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()

        if self.gt is not None:
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(
        self,
        epoch: int,
        current_iter: int,
    ) -> None:
        assert self.opt.path.models is not None
        assert self.opt.path.resume_models is not None

        if self.net_g_ema is not None:
            assert isinstance(self.net_g_ema.ema_model, nn.Module)
            self.save_network(
                self.net_g_ema.ema_model,
                "net_g_ema",
                self.opt.path.models,
                current_iter,
                "params_ema",
            )

            self.save_network(
                self.net_g, "net_g", self.opt.path.resume_models, current_iter, "params"
            )
        else:
            self.save_network(
                self.net_g, "net_g", self.opt.path.models, current_iter, "params"
            )

        if self.net_d is not None:
            self.save_network(
                self.net_d, "net_d", self.opt.path.resume_models, current_iter, "params"
            )

        self.save_training_state(epoch, current_iter)
