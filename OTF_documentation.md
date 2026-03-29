# Custom OTF Degradation Documentation

> Custom extensions to traiNNer-redux's OTF (on-the-fly) degradation pipeline.
> These features are **not in upstream** — they exist only in the WSL fork.

---

## Table of Contents

1. [Shared High-Frequency Noise](#1-shared-high-frequency-noise)
2. [Target GT Separation](#2-target-gt-separation)
3. [Paired + OTF Hybrid Training](#3-paired--otf-hybrid-training)
4. [Extended Compression](#4-extended-compression)
5. [Channel Shift](#5-channel-shift)
6. [Chroma Subsampling](#6-chroma-subsampling)
7. [Dithering](#7-dithering)
8. [Pipeline Order](#8-pipeline-order)
9. [YML Quick Reference](#9-yml-quick-reference)

---

## 1. High-Frequency Noise

Adds the **same** beta-distributed noise field to both GT and LQ after the final OTF crop. The model learns to preserve this shared texture/grain rather than treating it as noise to remove.

Optionally, GPU Non-Local Means denoising can be applied to LR **before the final resize** to remove existing noise, creating a cleaner input for the model to learn from.

### How It Works

1. A noise field is sampled from a `Beta(a, b)` distribution at GT resolution.
2. Optionally forced to grayscale (single channel broadcast to RGB).
3. Either **normalized** (zero-centered, unit-variance, clamped to ±3σ, scaled by α) or used **raw** as `(beta - 0.5) * 2 * α`.
4. The noise is added to GT only. LR is left untouched.
5. GT is clamped to [0, 1].
6. **(Optional)** If `otf_hf_noise_denoise_lq` is enabled, LR is denoised with GPU NLMeans **before the final resize** (earlier in the pipeline). This is applied when `otf_hf_noise_prob > 0`.

### Pipeline Position

```
... → Noise2 → ★ NLMeans denoise LQ (optional) ★ → Compress2/FinalResize/FinalSinc → Clamp → Crop → ★ HF noise (GT only) ★ → Dithering → Dequeue
```

### Config Fields

All fields are **top-level** (same level as `high_order_degradation`, not inside `datasets`).

```yaml
# ── Shared High-Frequency Noise ───────────────────────────────────────────────
# Adds identical noise to both GT and LQ so the model preserves texture/grain.

otf_hf_noise_prob: 0.0          # Probability of applying. 0 = disabled.
                                        # Type: float, range [0, 1]

otf_hf_noise_normalize: true     # true:  zero-center, normalize to unit variance,
                                        #        clamp ±3σ, then scale by alpha.
                                        # false: raw mode, noise = (beta - 0.5) * 2 * alpha.
                                        # Type: bool

otf_hf_noise_alpha_range: [0.01, 0.05]
                                        # Amplitude range. One alpha is sampled uniformly
                                        # per batch element from [min, max].
                                        # Type: [float, float]

otf_hf_noise_beta_shape_range: [2, 5]
                                        # Range for the Beta distribution 'a' parameter.
                                        # Also used for 'b' when beta_offset_range is null.
                                        # Type: [float, float]

otf_hf_noise_beta_offset_range: ~
                                        # Optional. When set, b = a + uniform(offset_min, offset_max)
                                        # instead of sampling b independently.
                                        # Use [1, 5] to approximate the external hf_noise snippet.
                                        # Type: [float, float] or null (~)

otf_hf_noise_gray_prob: 1.0      # Probability the noise is single-channel (grayscale),
                                        # broadcast to all RGB channels.
                                        # 1.0 = always grayscale, 0.0 = always per-channel color.
                                        # Type: float, range [0, 1]

otf_hf_noise_denoise_lq: false          # Apply GPU NLMeans denoising to LQ before final resize.
                                        # Removes existing noise for cleaner model input.
                                        # Only active when otf_hf_noise_prob > 0.
                                        # Type: bool

otf_hf_noise_denoise_strength: 30.0     # NLMeans filter strength (h parameter, 0-255 scale).
                                        # Higher = more smoothing. 10-20 is mild, 30-50 is strong.
                                        # Type: float
```

### Example: Enable with defaults

```yaml
otf_hf_noise_prob: 0.5
# All other fields use their defaults (shown above).
```

### Example: Stronger grain, color noise allowed

```yaml
otf_hf_noise_prob: 0.8
otf_hf_noise_alpha_range: [0.03, 0.10]
otf_hf_noise_beta_shape_range: [1.5, 4]
otf_hf_noise_gray_prob: 0.5        # 50% grayscale, 50% color noise
```

### Example: Approximate external hf_noise snippet style

```yaml
otf_hf_noise_prob: 0.7
otf_hf_noise_normalize: false       # raw (beta - 0.5) * 2 * alpha mode
otf_hf_noise_alpha_range: [0.02, 0.08]
otf_hf_noise_beta_shape_range: [2, 5]
otf_hf_noise_beta_offset_range: [1, 5]  # b = a + offset, skews distribution
```

### Example: HF noise with LQ denoising

```yaml
otf_hf_noise_prob: 0.6
otf_hf_noise_alpha_range: [0.02, 0.07]
otf_hf_noise_denoise_lq: true           # Denoise LQ before final resize
otf_hf_noise_denoise_strength: 25.0     # Moderate smoothing
```

---

## 2. Target GT Separation

Allows using **different HR images** as the supervision target (what the model trains toward) vs the source image that gets degraded by OTF to produce LR.

**Use case**: Your `dataroot_gt` contains the original images that go through the full OTF degradation pipeline to produce realistic LR. Your `target_dataroot_gt` contains processed/augmented versions of those same images (e.g., pre-sharpened, color-corrected, artifact-cleaned) that serve as the actual training target.

### How It Works

1. **Dataset** (`realesrgan_dataset.py`): Loads both `gt` (from `dataroot_gt`) and `target_gt` (from `target_dataroot_gt`). Same augmentation (flip/rotation) is applied to both. Same crop coordinates are used for both.
2. **Model** (`realesrgan_model.py`): OTF degrades `gt` (the source) to produce LQ. The model's supervision target `self.gt` is set to `target_gt`. Both are cropped at the same location using `paired_random_crop_multi_gt()`.

### Config Field

This field goes **inside `datasets: train:`**, not at top level.

```yaml
datasets:
  train:
    type: realesrgandataset
    dataroot_gt: [
      datasets/train/my_data/HR
    ]

    # ── Target GT Separation ──────────────────────────────────────────────────
    # Optional. When set, these images are used as supervision targets while
    # dataroot_gt images are used as the source for OTF LR synthesis.
    # Must have the same number of root folders as dataroot_gt.
    # Images must match dataroot_gt 1:1 in relative paths and dimensions.
    target_dataroot_gt: [
      datasets/train/my_data/HR_augmented
    ]
```

### Constraints

- `target_dataroot_gt` must have the **same number of folders** as `dataroot_gt`.
- Images must match **1:1 in relative paths** (same filenames in same subfolder structure).
- Images must have **identical dimensions** — a size mismatch raises `ValueError`.
- Not supported with **lmdb** datasets.
- When `target_dataroot_gt` is **null or omitted**, behavior is identical to upstream (GT is both source and target).

### Example: OTF source from originals, train toward augmented versions

```yaml
datasets:
  train:
    type: realesrgandataset
    dataroot_gt: [
      datasets/train/dataset5/HR
    ]
    target_dataroot_gt: [
      datasets/train/dataset5/HR_augmented
    ]
    # ... rest of dataset config ...
```

---

## 3. Paired + OTF Hybrid Training

Mixes **paired** (pre-generated LR/HR) and **OTF** (on-the-fly degraded) data within the same training run. Each batch randomly picks one source or the other.

### How It Works

1. **Dataset** (`realesrgan_paired_dataset.py`): Wraps both a `RealESRGANDataset` (OTF) and a `PairedImageDataset` (paired). Each `__getitem__` emits both, with keys prefixed `otf_*` and `paired_*`.
2. **Model** (`realesrgan_paired_model.py`): At each iteration, a coin flip (controlled by `dataroot_lq_prob`) decides whether to use the paired LR/HR or run the OTF degradation pipeline on the OTF HR.

### Activation

This mode activates automatically when **both** conditions are met:
- `high_order_degradation: true`
- `dataroot_lq_prob` > 0

The model registry in `__init__.py` checks these and selects `RealESRGANPairedModel`.

### Config Fields

`dataroot_lq_prob` is **top-level**. `paired_dataroot_gt` and `dataroot_lq` go inside **`datasets: train:`**.

```yaml
# ── Paired + OTF Hybrid ──────────────────────────────────────────────────────
# Top-level: probability of using paired data vs OTF per iteration.
dataroot_lq_prob: 0.3                   # 0.3 = 30% paired, 70% OTF.
                                        # Type: float, range [0, 1]
                                        # 0 = pure OTF (default, paired model not loaded)

datasets:
  train:
    type: realesrganpaireddataset        # NOTE: different dataset type!

    dataroot_gt: [
      datasets/train/my_data/HR          # Used by OTF sub-dataset for degradation source.
    ]                                    # Also used by paired sub-dataset if paired_dataroot_gt is null.

    dataroot_lq: [
      datasets/train/my_data/LR          # Pre-generated LR images for the paired sub-dataset.
    ]

    # ── Optional: separate HR for paired sub-dataset ──────────────────────────
    # When set, the paired sub-dataset uses this folder for HR instead of dataroot_gt.
    # Useful when your OTF source images differ from your paired HR images.
    paired_dataroot_gt: [
      datasets/train/my_data/HR_paired
    ]
    # Type: list[str] or null (~)
    # Default: null (paired sub-dataset uses dataroot_gt)
```

### Dataset type

When using hybrid mode, the dataset type **must** be `realesrganpaireddataset` (not `realesrgandataset`).

### Index wrapping

The paired sub-dataset wraps its index: `index % len(paired_dataset)`. This means the paired dataset can be smaller than the OTF dataset — it will cycle.

### Validation pass-through

The paired model detects validation batches (no `paired_`/`otf_` prefixed keys) and passes them directly to the parent `RealESRGANModel.feed_data()`.

### Example: 30% paired, 70% OTF, same HR source

```yaml
high_order_degradation: true
dataroot_lq_prob: 0.3

datasets:
  train:
    type: realesrganpaireddataset
    dataroot_gt: [
      datasets/train/dataset5/HR
    ]
    dataroot_lq: [
      datasets/train/dataset5/LR
    ]
    # paired_dataroot_gt not set → paired uses same dataroot_gt
    # ... rest of dataset config (blur kernels, etc.) ...
```

### Example: Separate HR folders for OTF vs paired

```yaml
high_order_degradation: true
dataroot_lq_prob: 0.5

datasets:
  train:
    type: realesrganpaireddataset
    dataroot_gt: [
      datasets/train/originals/HR
    ]
    dataroot_lq: [
      datasets/train/curated_pairs/LR
    ]
    paired_dataroot_gt: [
      datasets/train/curated_pairs/HR
    ]
    # ... rest of dataset config ...
```

---

## 4. Extended Compression

Extends the existing JPEG compression stages to support additional algorithms: **WebP**, **H.264**, **HEVC**, **MPEG-2**, **MPEG-4**, and **VP9**. Each compression stage independently selects one algorithm from a weighted list.

### How It Works

1. At each compression stage (stage 1 and stage 2), one algorithm is randomly chosen from `compress_algorithms` using `compress_algorithm_probs` as weights.
2. **JPEG** uses the existing GPU-based DiffJPEG (fast, differentiable).
3. **WebP** uses OpenCV encode/decode round-trip on CPU.
4. **Video codecs** (H.264, HEVC, MPEG-2, MPEG-4, VP9) use ffmpeg subprocess pipes on CPU — a single image is encoded as a video frame and decoded back. These are slower but produce authentic video compression artifacts.
5. Algorithms are **mutually exclusive** per stage — only one fires per iteration.

### Pipeline Position

Compression replaces the original JPEG stages in the pipeline:

```
... → Noise1 → ★ Compress1 ★ → Blur2 → ... → Noise2 → ★ Compress2 ★ / FinalResize / FinalSinc → ...
```

### Config Fields

All fields are **top-level**. Existing `jpeg_prob`/`jpeg_range`/`jpeg_prob2`/`jpeg_range2` are preserved for backward compatibility.

```yaml
# ── Stage 1 Compression ──────────────────────────────────────────────────────

jpeg_prob: 0.8                          # Probability of ANY compression in stage 1.
                                        # This controls whether compression fires at all.
                                        # Type: float, range [0, 1]

jpeg_range: [50, 95]                    # JPEG quality range (used when "jpeg" is selected).
                                        # Type: [int, int]

compress_algorithms: [jpeg]             # Algorithms to choose from. One is picked per iteration.
                                        # Options: jpeg, webp, h264, hevc, mpeg2, mpeg4, vp9
                                        # Type: list[str]
                                        # Default: [jpeg] (backward compatible — pure JPEG)

compress_algorithm_probs: [1.0]         # Selection weights. Must match length of compress_algorithms.
                                        # Type: list[float]

compress_webp_range: [50, 95]           # WebP quality range (1-100, higher = better).
                                        # Type: [int, int]

compress_h264_range: [20, 40]           # H.264 CRF range (0-51, lower = better quality).
                                        # Type: [int, int]

compress_hevc_range: [20, 40]           # HEVC CRF range (0-51, lower = better quality).
                                        # Type: [int, int]

compress_mpeg2_range: [2, 20]           # MPEG-2 qscale range (1-31, lower = better quality).
                                        # Type: [int, int]

compress_mpeg4_range: [2, 20]           # MPEG-4 qscale range (1-31, lower = better quality).
                                        # Type: [int, int]

compress_vp9_range: [20, 50]            # VP9 CRF range (0-63, lower = better quality).
                                        # Type: [int, int]

compress_video_sampling: [444, 422, 420]
                                        # Video chroma subsampling for video codecs.
                                        # One is randomly selected per video compression call.
                                        # Options: 444, 422, 420
                                        # Type: list[str]
```

Stage 2 uses the same field names with a `2` suffix: `compress_algorithms2`, `compress_algorithm_probs2`, `compress_webp_range2`, etc.

### Example: Pure JPEG (default, backward compatible)

```yaml
jpeg_prob: 0.8
jpeg_range: [50, 95]
# compress_algorithms defaults to [jpeg] — no change needed.
```

### Example: Mixed compression with JPEG, WebP, and HEVC

```yaml
# Stage 1: 50% JPEG, 30% HEVC, 20% WebP
jpeg_prob: 0.8
jpeg_range: [50, 95]
compress_algorithms: [jpeg, hevc, webp]
compress_algorithm_probs: [0.5, 0.3, 0.2]
compress_hevc_range: [20, 40]
compress_webp_range: [50, 95]
compress_video_sampling: [444, 422, 420]

# Stage 2: 70% JPEG, 30% WebP
jpeg_prob2: 0.8
jpeg_range2: [60, 95]
compress_algorithms2: [jpeg, webp]
compress_algorithm_probs2: [0.7, 0.3]
compress_webp_range2: [60, 95]
```

### Example: Video-codec-heavy pipeline (simulating video sources)

```yaml
jpeg_prob: 0.9
jpeg_range: [40, 90]
compress_algorithms: [jpeg, h264, hevc, mpeg2, mpeg4, vp9]
compress_algorithm_probs: [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]
compress_h264_range: [18, 35]
compress_hevc_range: [20, 38]
compress_mpeg2_range: [3, 15]
compress_mpeg4_range: [3, 15]
compress_vp9_range: [25, 45]
compress_video_sampling: [420, 422]
```

### Performance Notes

- **JPEG** (DiffJPEG): GPU, fast, ~0ms overhead.
- **WebP**: CPU, fast (~1-5ms per image via OpenCV).
- **Video codecs**: CPU + ffmpeg subprocess, slower (~10-50ms per image depending on resolution and codec). Each image is piped through an ffmpeg encode→decode round-trip. Use moderate probabilities if training speed is a concern.

### Dependencies

- **WebP**: OpenCV (already a traiNNer dependency).
- **Video codecs**: `ffmpeg` must be installed and accessible in `$PATH`. Install with `apt install ffmpeg` (Ubuntu/Debian) or equivalent for your OS.

---

## 5. Channel Shift

Applies per-channel spatial displacement to simulate color fringing, chromatic aberration, and chroma misalignment artifacts found in real-world images from cheap lenses, analog capture, or video processing.

### How It Works

1. A color space is randomly selected from `shift_types`.
2. The image is converted to that color space (RGB stays as-is, YUV uses BT.2020, CMYK uses pepeline).
3. Each channel is independently shifted by a random amount within its configured range.
4. The image is converted back to RGB.

### Pipeline Position

Applied before the first resize, after blur:

```
... → Blur1 → ★ Shift ★ → Subsampling → Resize1 → ...
```

### Config Fields

All fields are **top-level**.

```yaml
# ── Channel Shift ─────────────────────────────────────────────────────────────

shift_prob: 0.0                         # Probability of applying shift. 0 = disabled.
                                        # Type: float, range [0, 1]

shift_types: [rgb]                      # Color spaces to shift in. One is randomly selected.
                                        # Options: rgb, yuv, cmyk
                                        # Type: list[str]

shift_percent: false                    # true: amounts are percentages of image dimensions.
                                        # false: amounts are in pixels.
                                        # Type: bool

# Per-channel shift ranges: [[x_min, x_max], [y_min, y_max]]
# [0, 0] means no shift for that axis.

shift_rgb_r: [[0, 0], [0, 0]]          # R channel shift range.
shift_rgb_g: [[0, 0], [0, 0]]          # G channel shift range.
shift_rgb_b: [[0, 0], [0, 0]]          # B channel shift range.

shift_yuv_y: [[0, 0], [0, 0]]          # Y channel shift range (BT.2020 YCbCr).
shift_yuv_u: [[0, 0], [0, 0]]          # U (Cb) channel shift range.
shift_yuv_v: [[0, 0], [0, 0]]          # V (Cr) channel shift range.

shift_cmyk_c: [[0, 0], [0, 0]]         # C channel shift range (CMYK).
shift_cmyk_m: [[0, 0], [0, 0]]         # M channel shift range.
shift_cmyk_y: [[0, 0], [0, 0]]         # Y channel shift range.
shift_cmyk_k: [[0, 0], [0, 0]]         # K channel shift range.
```

### Example: Mild RGB chromatic aberration

```yaml
shift_prob: 0.3
shift_types: [rgb]
shift_rgb_r: [[-2, 2], [-2, 2]]
shift_rgb_g: [[0, 0], [0, 0]]          # Keep green stable (reference channel)
shift_rgb_b: [[-2, 2], [-2, 2]]
```

### Example: YUV chroma shift (mimics analog video)

```yaml
shift_prob: 0.4
shift_types: [yuv]
shift_yuv_y: [[0, 0], [0, 0]]          # Luma stays put
shift_yuv_u: [[-3, 3], [-1, 1]]        # Horizontal chroma shift
shift_yuv_v: [[-3, 3], [-1, 1]]
```

### Example: Mixed color space shifts

```yaml
shift_prob: 0.3
shift_types: [rgb, yuv]                # 50/50 chance of RGB or YUV shift
shift_rgb_r: [[-1, 1], [-1, 1]]
shift_rgb_b: [[-1, 1], [-1, 1]]
shift_yuv_u: [[-2, 2], [0, 0]]
shift_yuv_v: [[-2, 2], [0, 0]]
```

### Dependencies

- **RGB shift**: OpenCV only (already available).
- **YUV shift**: Requires `pepeline` (for BT.2020 YCbCr conversion).
- **CMYK shift**: Requires `pepeline` (for RGB↔CMYK conversion).

---

## 6. Chroma Subsampling

Simulates chroma subsampling artifacts (4:2:0, 4:2:2, etc.) by converting to YCbCr, downsampling then upsampling the chroma channels, and converting back. Optionally applies Gaussian blur to chroma channels.

### How It Works

1. The image is converted to YCbCr using the selected standard (BT.601, BT.709, BT.2020, or SMPTE-240M).
2. The chroma channels (Cb, Cr) are downsampled according to the subsampling format.
3. The chroma channels are upsampled back to original resolution.
4. Optionally, Gaussian blur is applied to the chroma channels.
5. The image is converted back to RGB.

### Pipeline Position

Applied before the first resize, after shift:

```
... → Blur1 → Shift → ★ Subsampling ★ → Resize1 → ...
```

### Config Fields

All fields are **top-level**.

```yaml
# ── Chroma Subsampling ────────────────────────────────────────────────────────

subsampling_prob: 0.0                   # Probability of applying subsampling. 0 = disabled.
                                        # Type: float, range [0, 1]

subsampling_down_algorithms: [nearest]  # Downscale algorithms to choose from.
                                        # Options: nearest, box, hermite, linear, lagrange,
                                        #   cubic_catrom, cubic_mitchell, cubic_bspline, lanczos, gauss
                                        # Type: list[str]

subsampling_up_algorithms: [nearest]    # Upscale algorithms to choose from.
                                        # Type: list[str]

subsampling_formats: ["4:4:4"]          # Chroma subsampling formats to choose from.
                                        # Options: 4:4:4, 4:2:2, 4:1:1, 4:2:0, 4:1:0,
                                        #   4:4:0, 4:2:1, 4:1:2, 4:1:3
                                        # Type: list[str]

subsampling_blur_range: ~               # Optional Gaussian blur sigma range for chroma channels.
                                        # Format: [min_sigma, max_sigma]. Null (~) to disable.
                                        # Type: [float, float] or null

subsampling_ycbcr_type: ["601"]         # YCbCr standards to choose from.
                                        # Options: 601, 709, 2020, 240
                                        # Type: list[str]
```

### Example: Standard 4:2:0 subsampling

```yaml
subsampling_prob: 0.4
subsampling_formats: ["4:2:0"]
subsampling_down_algorithms: [lanczos, linear]
subsampling_up_algorithms: [linear, nearest]
subsampling_ycbcr_type: ["601"]
```

### Example: Mixed subsampling with chroma blur

```yaml
subsampling_prob: 0.5
subsampling_formats: ["4:2:0", "4:2:2", "4:4:0"]
subsampling_down_algorithms: [lanczos, linear, cubic_catrom]
subsampling_up_algorithms: [linear, nearest, cubic_mitchell]
subsampling_blur_range: [0.5, 2.0]
subsampling_ycbcr_type: ["601", "709"]
```

### Subsampling Format Reference

| Format | Horizontal | Vertical | Description |
|--------|-----------|----------|-------------|
| 4:4:4 | Full | Full | No subsampling (passthrough) |
| 4:2:2 | Half | Full | Standard broadcast video |
| 4:2:0 | Half | Half | DVD, Blu-ray, most web video |
| 4:1:1 | Quarter | Full | DV NTSC |
| 4:4:0 | Full | Half | Uncommon |
| 4:1:0 | Quarter | Half | Aggressive subsampling |

### Dependencies

- Requires `colour` (for YCbCr conversion with configurable standards).
- Requires `chainner_ext` (for high-quality resize with various filter algorithms).
- Requires `pepeline` (for `fast_color_level` clamping).

---

## 7. Dithering

Applies color quantization and dithering algorithms to simulate reduced color depth artifacts found in GIFs, indexed-color images, and low-bitdepth displays.

### How It Works

1. A dithering algorithm is randomly selected from `dithering_types`.
2. A quantization level is sampled from `dithering_quantize_range`.
3. The selected algorithm reduces the image to the specified number of color levels, using the chosen dithering method to distribute quantization error.

### Dithering Algorithms

| Type | Method | Description |
|------|--------|-------------|
| `quantize` | Direct | Hard quantization, no error diffusion. Produces banding. |
| `floydsteinberg` | Error diffusion | Classic Floyd-Steinberg. Balanced quality. |
| `jarvisjudiceninke` | Error diffusion | Larger kernel, smoother but slower. |
| `stucki` | Error diffusion | Similar to Jarvis, slightly different distribution. |
| `atkinson` | Error diffusion | Lighter diffusion, retains more contrast. Used in early Macs. |
| `burkes` | Error diffusion | Fast variant of Stucki. |
| `sierra` | Error diffusion | Three-row Sierra filter. |
| `tworowsierra` | Error diffusion | Two-row variant, faster. |
| `sierralite` | Error diffusion | Minimal Sierra variant. |
| `order` | Ordered | Uses a threshold map (Bayer pattern). Produces regular patterns. |
| `riemersma` | Space-filling | Riemersma dithering along a Hilbert curve. |

### Pipeline Position

Applied as the **last degradation** in the pipeline, after crop:

```
... → Clamp → Crop → SharedHFNoise → ★ Dithering ★ → Dequeue → Training
```

This placement ensures dithering artifacts appear at the final LQ resolution and are not blurred or distorted by subsequent operations.

### Config Fields

All fields are **top-level**.

```yaml
# ── Dithering ─────────────────────────────────────────────────────────────────

dithering_prob: 0.0                     # Probability of applying dithering. 0 = disabled.
                                        # Type: float, range [0, 1]

dithering_types: [quantize]             # Algorithms to choose from. One is selected per iteration.
                                        # Options: quantize, floydsteinberg, jarvisjudiceninke,
                                        #   stucki, atkinson, burkes, sierra, tworowsierra,
                                        #   sierralite, order, riemersma
                                        # Type: list[str]

dithering_quantize_range: [2, 10]       # Range of quantization levels per channel.
                                        # Lower = more aggressive quantization (more visible).
                                        # Type: [int, int]

dithering_map_size: [4, 8]              # Map sizes for ordered dithering. One is randomly selected.
                                        # Larger = less visible pattern but coarser.
                                        # Type: list[int]

dithering_history_range: [10, 15]       # History length range for Riemersma dithering.
                                        # Type: [int, int]

dithering_ratio_range: [0.1, 0.9]       # Decay ratio range for Riemersma dithering.
                                        # Type: [float, float]
```

### Example: Light quantization with error diffusion

```yaml
dithering_prob: 0.2
dithering_types: [quantize, floydsteinberg, atkinson]
dithering_quantize_range: [8, 32]       # Mild — 8-32 levels per channel
```

### Example: Aggressive dithering (simulating GIF-like quality)

```yaml
dithering_prob: 0.4
dithering_types: [quantize, floydsteinberg, order]
dithering_quantize_range: [2, 8]        # Strong — 2-8 levels per channel
dithering_map_size: [2, 4, 8]
```

### Example: Full variety

```yaml
dithering_prob: 0.3
dithering_types: [quantize, floydsteinberg, stucki, atkinson, burkes, order, riemersma]
dithering_quantize_range: [4, 16]
dithering_map_size: [4, 8]
dithering_history_range: [8, 20]
dithering_ratio_range: [0.2, 0.8]
```

### Dependencies

- Requires `chainner_ext` (for `UniformQuantization`, `error_diffusion_dither`, `ordered_dither`, `riemersma_dither`, `quantize`).

---

## 8. Pipeline Order

The complete OTF degradation pipeline with all custom features:

```
GT Source
  │
  ├─ USM sharpening (optional)
  ├─ ThickLines filter (optional)
  ├─ Blur 1
  ├─ ★ Channel Shift (optional)
  ├─ ★ Chroma Subsampling (optional)
  ├─ Resize 1
  ├─ Noise 1 (Gaussian or Poisson)
  ├─ ★ Compression 1 (JPEG / WebP / H.264 / HEVC / MPEG-2 / MPEG-4 / VP9)
  ├─ Blur 2
  ├─ Resize 2
  ├─ Noise 2 (Gaussian or Poisson)
  ├─ ★ NLMeans denoise LQ (optional, before final resize)
  ├─ ★ Compression 2 + Final Resize + Final Sinc (interleaved)
  ├─ Clamp & Round
  ├─ Random Crop (paired with GT)
  ├─ ★ Shared HF Noise (GT only)
  ├─ ★ Dithering (LQ only, last degradation)
  ├─ Dequeue/Enqueue (diversity pool)
  └─ Training
```

★ = custom additions (not in upstream traiNNer-redux).

---

## 9. YML Quick Reference

### All custom fields at a glance

| Field | Level | Type | Default | Feature |
|---|---|---|---|---|
| `otf_hf_noise_prob` | top | float | `0` | HF Noise |
| `otf_hf_noise_normalize` | top | bool | `true` | HF Noise |
| `otf_hf_noise_alpha_range` | top | [float, float] | `[0.01, 0.05]` | HF Noise |
| `otf_hf_noise_beta_shape_range` | top | [float, float] | `[2, 5]` | HF Noise |
| `otf_hf_noise_beta_offset_range` | top | [float, float] \| null | `~` | HF Noise |
| `otf_hf_noise_gray_prob` | top | float | `1` | HF Noise |
| `otf_hf_noise_denoise_lq` | top | bool | `false` | HF Noise |
| `otf_hf_noise_denoise_strength` | top | float | `30.0` | HF Noise |
| `target_dataroot_gt` | datasets.train | list[str] \| null | `~` | Target GT |
| `paired_dataroot_gt` | datasets.train | list[str] \| null | `~` | Paired Hybrid |
| `dataroot_lq_prob` | top | float | `0` | Paired Hybrid |
| `compress_algorithms` | top | list[str] | `[jpeg]` | Extended Compression |
| `compress_algorithm_probs` | top | list[float] | `[1.0]` | Extended Compression |
| `compress_webp_range` | top | [int, int] | `[50, 95]` | Extended Compression |
| `compress_h264_range` | top | [int, int] | `[20, 40]` | Extended Compression |
| `compress_hevc_range` | top | [int, int] | `[20, 40]` | Extended Compression |
| `compress_mpeg2_range` | top | [int, int] | `[2, 20]` | Extended Compression |
| `compress_mpeg4_range` | top | [int, int] | `[2, 20]` | Extended Compression |
| `compress_vp9_range` | top | [int, int] | `[20, 50]` | Extended Compression |
| `compress_video_sampling` | top | list[str] | `[444, 422, 420]` | Extended Compression |
| `shift_prob` | top | float | `0` | Channel Shift |
| `shift_types` | top | list[str] | `[rgb]` | Channel Shift |
| `shift_percent` | top | bool | `false` | Channel Shift |
| `shift_rgb_r` / `g` / `b` | top | [[int,int],[int,int]] | `[[0,0],[0,0]]` | Channel Shift |
| `shift_yuv_y` / `u` / `v` | top | [[int,int],[int,int]] | `[[0,0],[0,0]]` | Channel Shift |
| `shift_cmyk_c` / `m` / `y` / `k` | top | [[int,int],[int,int]] | `[[0,0],[0,0]]` | Channel Shift |
| `subsampling_prob` | top | float | `0` | Chroma Subsampling |
| `subsampling_down_algorithms` | top | list[str] | `[nearest]` | Chroma Subsampling |
| `subsampling_up_algorithms` | top | list[str] | `[nearest]` | Chroma Subsampling |
| `subsampling_formats` | top | list[str] | `[4:4:4]` | Chroma Subsampling |
| `subsampling_blur_range` | top | [float, float] \| null | `~` | Chroma Subsampling |
| `subsampling_ycbcr_type` | top | list[str] | `[601]` | Chroma Subsampling |
| `dithering_prob` | top | float | `0` | Dithering |
| `dithering_types` | top | list[str] | `[quantize]` | Dithering |
| `dithering_quantize_range` | top | [int, int] | `[2, 10]` | Dithering |
| `dithering_map_size` | top | list[int] | `[4, 8]` | Dithering |
| `dithering_history_range` | top | [int, int] | `[10, 15]` | Dithering |
| `dithering_ratio_range` | top | [float, float] | `[0.1, 0.9]` | Dithering |

> **"top"** = same indentation level as `high_order_degradation`, `scale`, `queue_size`, etc.
> **"datasets.train"** = inside `datasets: train:` block.

### Combining features

All three features are independent and can be combined. For example, you can use target GT separation alongside shared HF noise in a pure OTF config:

```yaml
high_order_degradation: true

# Shared HF noise
otf_hf_noise_prob: 0.6
otf_hf_noise_alpha_range: [0.02, 0.07]

# Standard OTF degradation params
blur_prob: 0.5
# ... etc ...

datasets:
  train:
    type: realesrgandataset
    dataroot_gt: [
      datasets/train/data/HR
    ]
    target_dataroot_gt: [
      datasets/train/data/HR_augmented
    ]
    # ... rest of dataset config ...
```

Or use all three together:

```yaml
high_order_degradation: true
dataroot_lq_prob: 0.2

# Shared HF noise
otf_hf_noise_prob: 0.5
otf_hf_noise_alpha_range: [0.01, 0.05]

datasets:
  train:
    type: realesrganpaireddataset
    dataroot_gt: [
      datasets/train/data/HR
    ]
    target_dataroot_gt: [
      datasets/train/data/HR_target
    ]
    dataroot_lq: [
      datasets/train/data/LR
    ]
    paired_dataroot_gt: [
      datasets/train/data/HR_paired
    ]
    # ... rest of dataset config ...
```

---

## Source Files

| File | What was modified |
|---|---|
| `traiNNer/utils/redux_options.py` | Added `target_dataroot_gt`, `paired_dataroot_gt`, all `otf_hf_noise_*` fields, all `compress_*` fields, all `dithering_*` fields, all `shift_*` fields, all `subsampling_*` fields |
| `traiNNer/models/realesrgan_model.py` | Added shared HF noise methods, `_apply_compression()`, `_apply_shift()`, `_apply_subsampling()`, `_apply_dithering()`; modified `feed_data()` for target GT flow and new degradation pipeline |
| `traiNNer/models/realesrgan_paired_model.py` | Added validation data pass-through (prefix detection) |
| `traiNNer/data/otf_degradations.py` | **NEW** — CPU-based degradation functions: compression (WebP, video codecs), dithering, channel shift, chroma subsampling, tensor↔numpy bridge |
| `traiNNer/data/realesrgan_dataset.py` | Added target GT loading, paired augmentation, aligned cropping |
| `traiNNer/data/realesrgan_paired_dataset.py` | Added `paired_dataroot_gt` override, index wrapping |
| `traiNNer/data/transforms.py` | Added `single_crop_vips()`, `augment_vips_pair()`, `paired_random_crop_multi_gt()` |
