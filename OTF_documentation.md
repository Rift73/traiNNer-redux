# Custom OTF Degradation Documentation

> Custom extensions to traiNNer-redux's OTF (on-the-fly) degradation pipeline.
> These features are **not in upstream** — they exist only in the WSL fork.

---

## Table of Contents

1. [Shared High-Frequency Noise](#1-shared-high-frequency-noise)
2. [Target GT Separation](#2-target-gt-separation)
3. [Paired + OTF Hybrid Training](#3-paired--otf-hybrid-training)
4. [YML Quick Reference](#4-yml-quick-reference)

---

## 1. Shared High-Frequency Noise

Adds the **same** beta-distributed noise field to both GT and LQ after the final OTF crop. The model learns to preserve this shared texture/grain rather than treating it as noise to remove.

### How It Works

1. A noise field is sampled from a `Beta(a, b)` distribution at GT resolution.
2. Optionally forced to grayscale (single channel broadcast to RGB).
3. Either **normalized** (zero-centered, unit-variance, clamped to ±3σ, scaled by α) or used **raw** as `(beta - 0.5) * 2 * α`.
4. The noise is added to GT, then **nearest-exact downscaled** to LQ resolution and added to LQ.
5. Both GT and LQ are clamped to [0, 1].

### Pipeline Position

```
... → final resize → clamp/round → random crop → ★ shared HF noise ★ → dequeue/enqueue → training
```

### Config Fields

All fields are **top-level** (same level as `high_order_degradation`, not inside `datasets`).

```yaml
# ── Shared High-Frequency Noise ───────────────────────────────────────────────
# Adds identical noise to both GT and LQ so the model preserves texture/grain.

otf_shared_hf_noise_prob: 0.0          # Probability of applying. 0 = disabled.
                                        # Type: float, range [0, 1]

otf_shared_hf_noise_normalize: true     # true:  zero-center, normalize to unit variance,
                                        #        clamp ±3σ, then scale by alpha.
                                        # false: raw mode, noise = (beta - 0.5) * 2 * alpha.
                                        # Type: bool

otf_shared_hf_noise_alpha_range: [0.01, 0.05]
                                        # Amplitude range. One alpha is sampled uniformly
                                        # per batch element from [min, max].
                                        # Type: [float, float]

otf_shared_hf_noise_beta_shape_range: [2, 5]
                                        # Range for the Beta distribution 'a' parameter.
                                        # Also used for 'b' when beta_offset_range is null.
                                        # Type: [float, float]

otf_shared_hf_noise_beta_offset_range: ~
                                        # Optional. When set, b = a + uniform(offset_min, offset_max)
                                        # instead of sampling b independently.
                                        # Use [1, 5] to approximate the external hf_noise snippet.
                                        # Type: [float, float] or null (~)

otf_shared_hf_noise_gray_prob: 1.0      # Probability the noise is single-channel (grayscale),
                                        # broadcast to all RGB channels.
                                        # 1.0 = always grayscale, 0.0 = always per-channel color.
                                        # Type: float, range [0, 1]
```

### Example: Enable with defaults

```yaml
otf_shared_hf_noise_prob: 0.5
# All other fields use their defaults (shown above).
```

### Example: Stronger grain, color noise allowed

```yaml
otf_shared_hf_noise_prob: 0.8
otf_shared_hf_noise_alpha_range: [0.03, 0.10]
otf_shared_hf_noise_beta_shape_range: [1.5, 4]
otf_shared_hf_noise_gray_prob: 0.5        # 50% grayscale, 50% color noise
```

### Example: Approximate external hf_noise snippet style

```yaml
otf_shared_hf_noise_prob: 0.7
otf_shared_hf_noise_normalize: false       # raw (beta - 0.5) * 2 * alpha mode
otf_shared_hf_noise_alpha_range: [0.02, 0.08]
otf_shared_hf_noise_beta_shape_range: [2, 5]
otf_shared_hf_noise_beta_offset_range: [1, 5]  # b = a + offset, skews distribution
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

## 4. YML Quick Reference

### All custom fields at a glance

| Field | Level | Type | Default | Feature |
|---|---|---|---|---|
| `otf_shared_hf_noise_prob` | top | float | `0` | Shared HF Noise |
| `otf_shared_hf_noise_normalize` | top | bool | `true` | Shared HF Noise |
| `otf_shared_hf_noise_alpha_range` | top | [float, float] | `[0.01, 0.05]` | Shared HF Noise |
| `otf_shared_hf_noise_beta_shape_range` | top | [float, float] | `[2, 5]` | Shared HF Noise |
| `otf_shared_hf_noise_beta_offset_range` | top | [float, float] \| null | `~` | Shared HF Noise |
| `otf_shared_hf_noise_gray_prob` | top | float | `1` | Shared HF Noise |
| `target_dataroot_gt` | datasets.train | list[str] \| null | `~` | Target GT |
| `paired_dataroot_gt` | datasets.train | list[str] \| null | `~` | Paired Hybrid |
| `dataroot_lq_prob` | top | float | `0` | Paired Hybrid |

> **"top"** = same indentation level as `high_order_degradation`, `scale`, `queue_size`, etc.
> **"datasets.train"** = inside `datasets: train:` block.

### Combining features

All three features are independent and can be combined. For example, you can use target GT separation alongside shared HF noise in a pure OTF config:

```yaml
high_order_degradation: true

# Shared HF noise
otf_shared_hf_noise_prob: 0.6
otf_shared_hf_noise_alpha_range: [0.02, 0.07]

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
otf_shared_hf_noise_prob: 0.5
otf_shared_hf_noise_alpha_range: [0.01, 0.05]

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
| `traiNNer/utils/redux_options.py` | Added `target_dataroot_gt`, `paired_dataroot_gt`, all `otf_shared_hf_noise_*` fields |
| `traiNNer/models/realesrgan_model.py` | Added `_sample_beta_noise()`, `_generate_shared_hf_noise()`, `_apply_shared_hf_noise()`; modified `feed_data()` for target GT flow |
| `traiNNer/models/realesrgan_paired_model.py` | Added validation data pass-through (prefix detection) |
| `traiNNer/data/realesrgan_dataset.py` | Added target GT loading, paired augmentation, aligned cropping |
| `traiNNer/data/realesrgan_paired_dataset.py` | Added `paired_dataroot_gt` override, index wrapping |
| `traiNNer/data/transforms.py` | Added `single_crop_vips()`, `augment_vips_pair()`, `paired_random_crop_multi_gt()` |
