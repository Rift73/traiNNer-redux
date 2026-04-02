import os
import random
import sys
from os import path as osp

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    resize_pt,
)
from traiNNer.data.compress_video_batch import compress_video_batch
from traiNNer.data.gpu_degradations import (
    channel_shift_pt,
    chroma_subsample_pt,
    composite_rainbow_pt,
    detail_mask_neo_pt,
    interlace_pt,
    lowpass_filter_pt,
    ntsc_composite_pt,
    ordered_dither_pt,
    overshoot_pt,
    quantize_pt,
    scanline_pt,
    temporal_ghosting_pt,
)
from traiNNer.data.otf_degradations import (
    apply_dithering,
    apply_per_image,
    apply_per_image_parallel,
    compress_webp,
    nlmeans_denoise_pt,
)
from traiNNer.data.transforms import paired_random_crop, paired_random_crop_multi_gt
from traiNNer.models.sr_model import SRModel
from traiNNer.utils import RNG, DiffJPEG, get_root_logger
from traiNNer.utils.img_process_util import USMSharp, filter2d
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.registry import MODEL_REGISTRY
from traiNNer.utils.types import DataFeed

OTF_DEBUG_PATH = osp.abspath(
    osp.abspath(osp.join(osp.join(sys.argv[0], osp.pardir), "./debug/otf"))
)

ANTIALIAS_MODES = {"bicubic", "bilinear"}


@MODEL_REGISTRY.register(suffix="traiNNer")
class RealESRGANModel(SRModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        self.queue_lr: Tensor | None = None
        self.queue_gt: Tensor | None = None
        self.queue_ptr = 0
        self.kernel1: Tensor | None = None
        self.kernel2: Tensor | None = None
        self.sinc_kernel: Tensor | None = None

        self.jpeger = DiffJPEG(
            differentiable=False
        ).cuda()  # simulate JPEG compression artifacts
        self.queue_size = opt.queue_size

        self.thicklines = None
        if self.opt.thicklines_prob > 0:
            self.thicklines = ThickLines().cuda()

        self.otf_debug = opt.high_order_degradations_debug
        self.otf_debug_limit = opt.high_order_degradations_debug_limit

        if self.otf_debug:
            logger = get_root_logger()
            logger.info(
                "OTF debugging enabled. LR tiles will be saved to: %s", OTF_DEBUG_PATH
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self) -> None:
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """

        assert self.lq is not None
        assert self.gt is not None

        # initialize
        b, c, h, w = self.lq.size()
        if self.queue_lr is None:
            assert self.queue_size % b == 0, (
                f"queue size {self.queue_size} should be divisible by batch size {b}"
            )
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            assert self.queue_lr is not None
            assert self.queue_gt is not None
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            assert self.queue_lr is not None
            assert self.queue_gt is not None

            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.lq.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            self.queue_ptr = self.queue_ptr + b

    def _sample_beta_noise(
        self,
        channels: int,
        height: int,
        width: int,
        shape_a: Tensor,
        shape_b: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        distribution = torch.distributions.Beta(
            shape_a.view(-1, 1, 1, 1).expand(-1, channels, height, width),
            shape_b.view(-1, 1, 1, 1).expand(-1, channels, height, width),
        )
        return distribution.sample().to(device=device, dtype=dtype)

    def _generate_shared_hf_noise(self, img: Tensor) -> Tensor:
        batch_size, channels, height, width = img.shape
        rng = RNG.get_rng()
        sample_dtype = torch.float32
        device = img.device

        shape_a = torch.as_tensor(
            rng.uniform(*self.opt.otf_hf_noise_beta_shape_range, size=batch_size),
            device=device,
            dtype=sample_dtype,
        )
        if self.opt.otf_hf_noise_beta_offset_range is not None:
            shape_b = shape_a + torch.as_tensor(
                rng.uniform(
                    *self.opt.otf_hf_noise_beta_offset_range, size=batch_size
                ),
                device=device,
                dtype=sample_dtype,
            )
        else:
            shape_b = torch.as_tensor(
                rng.uniform(
                    *self.opt.otf_hf_noise_beta_shape_range, size=batch_size
                ),
                device=device,
                dtype=sample_dtype,
            )
        alpha = torch.as_tensor(
            rng.uniform(*self.opt.otf_hf_noise_alpha_range, size=batch_size),
            device=device,
            dtype=sample_dtype,
        ).view(-1, 1, 1, 1)

        noise = self._sample_beta_noise(
            channels,
            height,
            width,
            shape_a,
            shape_b,
            sample_dtype,
            device,
        )

        if channels > 1 and self.opt.otf_hf_noise_gray_prob > 0:
            gray_mask = torch.as_tensor(
                rng.uniform(size=batch_size) < self.opt.otf_hf_noise_gray_prob,
                device=device,
                dtype=torch.bool,
            )
            gray_count = int(gray_mask.sum().item())
            if gray_count > 0:
                gray_noise = self._sample_beta_noise(
                    1,
                    height,
                    width,
                    shape_a[gray_mask],
                    shape_b[gray_mask],
                    sample_dtype,
                    device,
                ).expand(-1, channels, -1, -1)
                noise[gray_mask] = gray_noise

        if self.opt.otf_hf_noise_normalize:
            noise = noise - noise.mean(dim=(1, 2, 3), keepdim=True)
            noise = noise / (noise.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
            noise = torch.clamp(noise, -3, 3) * alpha
        else:
            noise = (noise - 0.5) * (2 * alpha)

        return noise.to(dtype=img.dtype)

    def _apply_shared_hf_noise(self) -> None:
        if self.opt.otf_hf_noise_prob <= 0:
            return

        assert self.gt is not None
        assert self.lq is not None

        if RNG.get_rng().uniform() >= self.opt.otf_hf_noise_prob:
            return

        shared_noise = self._generate_shared_hf_noise(self.gt)
        self.gt = torch.clamp(self.gt + shared_noise, 0, 1)
        shared_noise_lq = resize_pt(
            shared_noise,
            size=self.lq.shape[-2:],
            mode="nearest-exact",
        )
        self.lq = torch.clamp(self.lq + shared_noise_lq, 0, 1)

    def _select_active_degradations(self) -> set[str]:
        """Select which custom degradations are active for this iteration.

        Uses the combo system if configured, otherwise all degradations
        are eligible (backward-compatible default behavior).

        Returns:
            Set of active degradation names. A degradation's probability
            check is only reached if its name is in this set.
        """
        combos = self.opt.otf_degradation_combos
        if combos is None:
            # No combo system configured — all degradations eligible
            return {
                "ntsc", "rainbow", "ghosting", "interlace", "scanline",
                "lowpass", "overshoot", "shift", "subsampling", "dithering",
                "compress", "hf_noise", "nlmeans",
            }

        active: set[str] = set()

        # Always add global degradations
        if self.opt.otf_global_degradations is not None:
            active.update(self.opt.otf_global_degradations)

        # Two-step selection: first check if any combo fires, then pick which
        no_combo_prob = self.opt.otf_no_combo_weight
        if RNG.get_rng().uniform() >= no_combo_prob:
            # A combo fires — pick which one using relative weights
            weights = self.opt.otf_degradation_combo_weights
            if weights is None:
                weights = [1.0] * len(combos)

            selected = random.choices(combos, weights=weights, k=1)[0]
            active.update(selected)

        return active

    def _apply_compression(self, out: Tensor, stage: int = 1) -> Tensor:
        """Apply a randomly chosen compression algorithm.

        JPEG: GPU DiffJPEG (unchanged).
        WebP: Pillow method=0 via parallel ThreadPool (1.8x faster than cv2).
        Video codecs: TorchCodec/PyAV batch encoding (9-48x faster than subprocess).
        """
        algos = (
            self.opt.compress_algorithms
            if stage == 1
            else self.opt.compress_algorithms2
        )
        probs = (
            self.opt.compress_algorithm_probs
            if stage == 1
            else self.opt.compress_algorithm_probs2
        )
        algo = random.choices(algos, probs)[0]

        if algo == "jpeg":
            jpeg_range = self.opt.compress_jpeg_range if stage == 1 else self.opt.compress_jpeg_range2
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        elif algo == "webp":
            quality_range = (
                self.opt.compress_webp_range
                if stage == 1
                else self.opt.compress_webp_range2
            )
            quality = int(RNG.get_rng().integers(*quality_range))
            out = torch.clamp(out, 0, 1)
            out = apply_per_image_parallel(
                out, lambda img: compress_webp(img, quality)
            )
        elif algo in ("h264", "hevc", "mpeg2", "mpeg4", "vp9"):
            range_attr = f"compress_{algo}_range{'2' if stage == 2 else ''}"
            quality_range = getattr(self.opt, range_attr)
            quality = int(RNG.get_rng().integers(*quality_range))
            video_sampling = (
                self.opt.compress_video_sampling
                if stage == 1
                else self.opt.compress_video_sampling2
            )
            out = torch.clamp(out, 0, 1)
            out = compress_video_batch(out, algo, quality, video_sampling)
        return out

    def _apply_shift(self, out: Tensor) -> Tensor:
        """Apply channel shift degradation (GPU, no apply_per_image)."""
        rng = RNG.get_rng()
        shift_type = rng.choice(self.opt.shift_types)

        if shift_type == "rgb":
            amounts = [
                self.opt.shift_rgb_r,
                self.opt.shift_rgb_g,
                self.opt.shift_rgb_b,
            ]
        elif shift_type == "yuv":
            amounts = [
                self.opt.shift_yuv_y,
                self.opt.shift_yuv_u,
                self.opt.shift_yuv_v,
            ]
        elif shift_type == "cmyk":
            amounts = [
                self.opt.shift_cmyk_c,
                self.opt.shift_cmyk_m,
                self.opt.shift_cmyk_y,
                self.opt.shift_cmyk_k,
            ]
        else:
            return out

        return channel_shift_pt(out, shift_type, amounts, self.opt.shift_percent)

    def _apply_subsampling(self, out: Tensor) -> Tensor:
        """Apply chroma subsampling degradation (GPU, no apply_per_image)."""
        rng = RNG.get_rng()
        down_alg = rng.choice(self.opt.subsampling_down_algorithms)
        up_alg = rng.choice(self.opt.subsampling_up_algorithms)
        fmt = rng.choice(self.opt.subsampling_formats)
        ycbcr = rng.choice(self.opt.subsampling_ycbcr_type)
        blur_sigma = None
        if self.opt.subsampling_blur_range is not None:
            blur_sigma = float(rng.uniform(*self.opt.subsampling_blur_range))

        return chroma_subsample_pt(out, down_alg, up_alg, fmt, blur_sigma, ycbcr)

    def _apply_dithering(self, out: Tensor) -> Tensor:
        """Apply dithering degradation to the LQ tensor.

        GPU path for quantize/ordered dither (5-10x faster).
        CPU path via chainner_ext for error diffusion and riemersma
        (causal scan dependency prevents efficient GPU parallelization).
        """
        rng = RNG.get_rng()
        dtype = rng.choice(self.opt.dithering_types)
        quantize_ch = int(rng.integers(*self.opt.dithering_quantize_range))

        if dtype == "quantize":
            return quantize_pt(out, quantize_ch)
        elif dtype == "order":
            map_size = int(rng.choice(self.opt.dithering_map_size))
            return ordered_dither_pt(out, quantize_ch, map_size)
        else:
            # Error diffusion / riemersma — CPU path (chainner_ext)
            map_size = int(rng.choice(self.opt.dithering_map_size))
            history = int(rng.integers(*self.opt.dithering_history_range))
            decay_ratio = float(rng.uniform(*self.opt.dithering_ratio_range))
            return apply_per_image(
                out,
                lambda img: apply_dithering(
                    img, dtype, quantize_ch, map_size, history, decay_ratio
                ),
            )

    def _apply_ghosting(self, out: Tensor) -> Tensor:
        """Apply temporal ghosting degradation (GPU)."""
        rng = RNG.get_rng()
        sx = int(rng.integers(*self.opt.ghosting_shift_x))
        sy = int(rng.integers(*self.opt.ghosting_shift_y))
        opacity = float(rng.uniform(*self.opt.ghosting_opacity))
        return temporal_ghosting_pt(out, sx, sy, opacity)

    def _apply_interlace(self, out: Tensor) -> Tensor:
        """Apply interlace combing degradation (GPU)."""
        rng = RNG.get_rng()
        shift = int(rng.integers(*self.opt.interlace_field_shift))
        field = rng.choice(self.opt.interlace_dominant_field)
        return interlace_pt(out, shift, field)

    def _apply_lowpass(self, out: Tensor) -> Tensor:
        """Apply Butterworth lowpass filter degradation (GPU).

        Optionally preserves detail/edges using a mask computed from HQ.
        """
        rng = RNG.get_rng()
        cutoff = float(rng.uniform(*self.opt.lowpass_cutoff))
        order = int(rng.integers(*self.opt.lowpass_order))

        filtered = lowpass_filter_pt(out, cutoff, order)

        if self.opt.lowpass_detail_mask:
            mask = detail_mask_neo_pt(
                self.gt, lines_brz=self.opt.lowpass_mask_lines_brz
            )
            # Resize mask to match LQ if sizes differ
            if mask.shape[2:] != out.shape[2:]:
                mask = F.interpolate(
                    mask, size=out.shape[2:], mode="bilinear", align_corners=False
                )
            filtered = mask * out + (1.0 - mask) * filtered

        return filtered

    def _apply_ntsc(self, out: Tensor) -> Tensor:
        """Apply full NTSC composite simulation (GPU)."""
        _NTSC_PRESETS = {
            "broadcast": (False, 4.2, 500, 0.03, 0.0, 0.0),
            "vhs_sp": (True, 3.0, 500, 0.06, 0.03, 0.8),
            "vhs_ep": (True, 1.6, 300, 0.10, 0.06, 0.5),
        }

        rng = RNG.get_rng()
        preset = rng.choice(self.opt.ntsc_preset)
        enable_vhs, vlbw, cubw, n, ln, er = _NTSC_PRESETS.get(
            preset, _NTSC_PRESETS["broadcast"]
        )

        # Override with user config or preset defaults
        if self.opt.ntsc_enable_vhs:
            enable_vhs = True

        noise_val = float(rng.uniform(*self.opt.ntsc_noise)) if self.opt.ntsc_noise != [n, n] else n
        luma_noise_val = float(rng.uniform(*self.opt.ntsc_luma_noise)) if self.opt.ntsc_luma_noise != [ln, ln] else ln
        ghost_amp = float(rng.uniform(*self.opt.ntsc_ghost_amplitude))
        ghost_del = float(rng.uniform(*self.opt.ntsc_ghost_delay_us))
        ghost_ph = float(rng.uniform(*self.opt.ntsc_ghost_phase))
        jitter_val = float(rng.uniform(*self.opt.ntsc_jitter))
        ringing_val = float(rng.uniform(*self.opt.ntsc_edge_ringing)) if self.opt.ntsc_edge_ringing != [er, er] else er
        vlbw_val = float(rng.uniform(*self.opt.ntsc_vhs_luma_bw)) if self.opt.ntsc_vhs_luma_bw != [vlbw, vlbw] else vlbw
        cubw_val = float(rng.uniform(*self.opt.ntsc_color_under_bw)) if self.opt.ntsc_color_under_bw != [cubw, cubw] else cubw
        trail_val = float(rng.uniform(*self.opt.ntsc_tape_trailing))
        intensity_val = float(rng.uniform(*self.opt.ntsc_intensity))

        return ntsc_composite_pt(
            out,
            noise=noise_val,
            luma_noise=luma_noise_val,
            ghost_amplitude=ghost_amp,
            ghost_delay_us=ghost_del,
            ghost_phase=ghost_ph,
            jitter=jitter_val,
            edge_ringing=ringing_val,
            vhs_luma_bw=vlbw_val,
            color_under_bw=cubw_val,
            tape_trailing=trail_val,
            intensity=intensity_val,
            comb_mode=self.opt.ntsc_comb_mode,
            enable_vhs=enable_vhs,
        )

    def _apply_overshoot(self, out: Tensor) -> Tensor:
        """Apply edge overshoot/undershoot degradation (GPU)."""
        rng = RNG.get_rng()
        amount = float(rng.uniform(*self.opt.overshoot_amount))
        cutoff = float(rng.uniform(*self.opt.overshoot_cutoff))
        order = int(rng.integers(*self.opt.overshoot_order))
        return overshoot_pt(out, amount, cutoff, order)

    def _apply_rainbow(self, out: Tensor) -> Tensor:
        """Apply composite rainbow artifact (GPU)."""
        rng = RNG.get_rng()
        freq = float(rng.uniform(*self.opt.rainbow_subcarrier_freq))
        bw = float(rng.uniform(*self.opt.rainbow_chroma_bandwidth))
        intensity = float(rng.uniform(*self.opt.rainbow_intensity))
        phase_offset = float(rng.uniform(0, 2 * 3.141592653589793))
        return composite_rainbow_pt(
            out, freq, bw, intensity, self.opt.rainbow_phase_alternation, phase_offset
        )

    def _apply_scanline(self, out: Tensor) -> Tensor:
        """Apply CRT scanline darkening (GPU)."""
        rng = RNG.get_rng()
        strength = float(rng.uniform(*self.opt.scanline_strength))
        return scanline_pt(out, strength, self.opt.scanline_even_lines)

    @torch.no_grad()
    def feed_data(self, data: DataFeed) -> None:
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        if self.is_train:
            assert (
                "gt" in data
                and "kernel1" in data
                and "kernel2" in data
                and "sinc_kernel" in data
            )
            # training data synthesis
            gt_source = data["gt"]
            if gt_source.device != self.device:
                gt_source = gt_source.to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )
            target_gt = data.get("target_gt")
            if target_gt is not None and target_gt.device != self.device:
                target_gt = target_gt.to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )

            self.gt = target_gt if target_gt is not None else gt_source

            kernel1 = data["kernel1"]
            if kernel1.device != self.device:
                kernel1 = kernel1.to(
                    self.device,
                    non_blocking=True,
                )
            self.kernel1 = kernel1

            kernel2 = data["kernel2"]
            if kernel2.device != self.device:
                kernel2 = kernel2.to(
                    self.device,
                    non_blocking=True,
                )
            self.kernel2 = kernel2

            sinc_kernel = data["sinc_kernel"]
            if sinc_kernel.device != self.device:
                sinc_kernel = sinc_kernel.to(
                    self.device,
                    non_blocking=True,
                )
            self.sinc_kernel = sinc_kernel

            ori_h, ori_w = gt_source.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            if self.opt.lq_usm:
                usm_sharpener = USMSharp(
                    RNG.get_rng().integers(
                        *self.opt.lq_usm_radius_range, dtype=int, endpoint=True
                    )
                ).to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765
                out = usm_sharpener(gt_source)
            else:
                out = gt_source

            # thick lines
            if RNG.get_rng().uniform() < self.opt.thicklines_prob:
                assert self.thicklines is not None
                out = self.thicklines(out)

            # ── Select which custom degradations are active this iteration ──
            active = self._select_active_degradations()

            # ── NLMeans denoise source (before all degradations) ──
            if "nlmeans" in active and self.opt.otf_hf_noise_denoise_lq and self.opt.otf_hf_noise_prob > 0:
                out = nlmeans_denoise_pt(
                    out, h=self.opt.otf_hf_noise_denoise_strength
                )

            # ── Source-level artifacts (before blur1) ──
            if "ntsc" in active and RNG.get_rng().uniform() < self.opt.ntsc_prob:
                out = self._apply_ntsc(out)
            if "rainbow" in active and RNG.get_rng().uniform() < self.opt.rainbow_prob:
                out = self._apply_rainbow(out)
            if "ghosting" in active and RNG.get_rng().uniform() < self.opt.ghosting_prob:
                out = self._apply_ghosting(out)
            if "interlace" in active and RNG.get_rng().uniform() < self.opt.interlace_prob:
                out = self._apply_interlace(out)
            if "scanline" in active and RNG.get_rng().uniform() < self.opt.scanline_prob:
                out = self._apply_scanline(out)

            # blur
            if RNG.get_rng().uniform() < self.opt.blur_prob:
                out = filter2d(out, self.kernel1)

            # ── Mastering artifacts (after blur1, before shift) ──
            if "lowpass" in active and RNG.get_rng().uniform() < self.opt.lowpass_prob:
                out = self._apply_lowpass(out)
            if "overshoot" in active and RNG.get_rng().uniform() < self.opt.overshoot_prob:
                out = self._apply_overshoot(out)

            # channel shift (before resize1)
            if "shift" in active and RNG.get_rng().uniform() < self.opt.shift_prob:
                out = self._apply_shift(out)

            # chroma subsampling (before resize1)
            if "subsampling" in active and RNG.get_rng().uniform() < self.opt.subsampling_prob:
                out = self._apply_subsampling(out)

            # random resize
            updown_type = random.choices(["up", "down", "keep"], self.opt.resize_prob)[
                0
            ]
            if updown_type == "up":
                scale = RNG.get_rng().uniform(1, self.opt.resize_range[1])
            elif updown_type == "down":
                scale = RNG.get_rng().uniform(self.opt.resize_range[0], 1)
            else:
                scale = 1

            if scale != 1:
                assert len(self.opt.resize_mode_list) == len(
                    self.opt.resize_mode_prob
                ), "resize_mode_list and resize_mode_prob must be the same length"
                mode = random.choices(
                    self.opt.resize_mode_list, weights=self.opt.resize_mode_prob
                )[0]

                out = resize_pt(
                    out,
                    scale_factor=scale,
                    mode=mode,
                )

            # add noise
            gray_noise_prob = self.opt.gray_noise_prob
            if RNG.get_rng().uniform() < self.opt.gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt.noise_range,
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt.poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )
            # compression (stage 1: JPEG, WebP, or video codec)
            if "compress" in active and RNG.get_rng().uniform() < self.opt.compress_prob:
                out = self._apply_compression(out, stage=1)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if RNG.get_rng().uniform() < self.opt.blur_prob2:
                out = filter2d(out, self.kernel2)
            # random resize
            updown_type = random.choices(["up", "down", "keep"], self.opt.resize_prob2)[
                0
            ]
            if updown_type == "up":
                scale = RNG.get_rng().uniform(1, self.opt.resize_range2[1])
            elif updown_type == "down":
                scale = RNG.get_rng().uniform(self.opt.resize_range2[0], 1)
            else:
                scale = 1

            if scale != 1:
                assert len(self.opt.resize_mode_list2) == len(
                    self.opt.resize_mode_prob2
                ), "resize_mode_list2 and resize_mode_prob2 must be the same length"
                mode = random.choices(
                    self.opt.resize_mode_list2, weights=self.opt.resize_mode_prob2
                )[0]
                out = resize_pt(
                    out,
                    size=(
                        int(ori_h / self.opt.scale * scale),
                        int(ori_w / self.opt.scale * scale),
                    ),
                    mode=mode,
                )
            # add noise
            gray_noise_prob = self.opt.gray_noise_prob2
            if RNG.get_rng().uniform() < self.opt.gaussian_noise_prob2:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt.noise_range2,
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt.poisson_scale_range2,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

            # Compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + compression
            #   2. compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + compression + Resize) will introduce twisted lines.

            assert len(self.opt.resize_mode_list3) == len(self.opt.resize_mode_prob3), (
                "resize_mode_list3 and resize_mode_prob3 must be the same length"
            )

            mode = random.choices(
                self.opt.resize_mode_list3, weights=self.opt.resize_mode_prob3
            )[0]
            if RNG.get_rng().uniform() < 0.5:
                # resize back + the final sinc filter
                out = resize_pt(
                    out,
                    size=(ori_h // self.opt.scale, ori_w // self.opt.scale),
                    mode=mode,
                )
                out = filter2d(out, self.sinc_kernel)
                # compression (stage 2)
                if "compress" in active and RNG.get_rng().uniform() < self.opt.compress_prob2:
                    out = self._apply_compression(out, stage=2)
            else:
                # compression (stage 2)
                if "compress" in active and RNG.get_rng().uniform() < self.opt.compress_prob2:
                    out = self._apply_compression(out, stage=2)
                # resize back + the final sinc filter
                out = resize_pt(
                    out,
                    size=(ori_h // self.opt.scale, ori_w // self.opt.scale),
                    mode=mode,
                )
                out = filter2d(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # random crop
            gt_size = self.opt.datasets["train"].gt_size
            assert gt_size is not None
            if target_gt is not None:
                cropped_gts, self.lq = paired_random_crop_multi_gt(
                    [self.gt, gt_source], self.lq, gt_size, self.opt.scale
                )
                self.gt = cropped_gts[0]
            else:
                self.gt, self.lq = paired_random_crop(
                    self.gt, self.lq, gt_size, self.opt.scale
                )
            if "hf_noise" in active:
                self._apply_shared_hf_noise()

            # dithering (last degradation, applied to LQ only)
            if "dithering" in active and RNG.get_rng().uniform() < self.opt.dithering_prob:
                self.lq = self._apply_dithering(self.lq)

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            i = 1
            if self.otf_debug:
                os.makedirs(OTF_DEBUG_PATH, exist_ok=True)
                while os.path.exists(rf"{OTF_DEBUG_PATH}/{i:06d}_otf_lq.png"):
                    i += 1

                if i <= self.otf_debug_limit or self.otf_debug_limit == 0:
                    torchvision.utils.save_image(
                        self.lq,
                        os.path.join(OTF_DEBUG_PATH, f"{i:06d}_otf_lq.png"),
                        padding=0,
                    )

                    torchvision.utils.save_image(
                        self.gt,
                        os.path.join(OTF_DEBUG_PATH, f"{i:06d}_otf_gt.png"),
                        padding=0,
                    )

            # moa
            if self.is_train and self.batch_augment:
                self.gt, self.lq = self.batch_augment(self.gt, self.lq)
        else:
            # for paired training or validation
            assert "lq" in data
            self.lq = data["lq"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )
            if "gt" in data:
                self.gt = data["gt"].to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )


class ThickLines(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, original: Tensor) -> Tensor:
        # Calculate the padding needed for the convolution
        pad = self.kernel_size // 2

        # Apply a max pooling operation with the same kernel size to both images
        min_original = -F.max_pool2d(-original, self.kernel_size, padding=pad, stride=1)
        avg_original = original * 3 / 4 + min_original * 1 / 4

        return avg_original

    def blend(self, original: Tensor, usm: Tensor) -> Tensor:
        input = torch.cat((original, usm), dim=1)
        return self.conv(input)  # pyright: ignore[reportCallIssue]
