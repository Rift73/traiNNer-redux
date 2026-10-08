# OTF degradation performance pass (2026-10-09)

Owner brief: squeeze the OTF degradation pipeline toward its limit with C/C++/CUDA where it pays, portable for other
people's machines (no `-march`, no machine-specific tuning, JIT-built kernels with the CPU path as fallback),
backend-only: the YAML contract is unchanged. Measured on the real loader and model, not on synthetic tensors.

## Contract (phase 1)

1. Work tree: a clone of the clean repo (`otf-cuda-dither` branch); the clean repo is untouched until the owner
   merges. 2. Kernels compile on any sm_75+ GPU through torch's `cpp_extension` on first use and are cached; without
   a CUDA toolkit the previous CPU path runs unchanged. 3. Outputs bit-identical to the previous code unless a table
   below says otherwise. 4. Benchmarks: one paired run per case, 50 iterations, median, CUDA-synced. 5. Out of
   scope: codec backends (owner's own benchmarks settled them), palette-mode dithering on the GPU (phase 2),
   CUDA-graph capture and side-stream pipelining of the chain (phase 2).

## Baseline (benchmarks/otf_profile.py, real RealESRGANDataset + RealESRGANModel.feed_data, batch 8, GT 256)

| run | feed_data median / p90 | fetch | hidden host syncs |
|---|---|---|---|
| Floyd-Steinberg, JPEG | 21.2 / 24.5 ms | 0.6 ms | ~55 per batch |
| Riemersma, JPEG | 21.5 / 25.4 ms | 0.7 ms | |
| Floyd-Steinberg, mixed codecs | 31.1 / 68.7 ms | 0.6 ms | |

Where the time went (Floyd-Steinberg/JPEG): Poisson noise 5.45 ms (23 %, computed then multiplied by 0 when the
range is [0, 0], 16 syncs); DiffJPEG 4.16 ms (20 %, per-image Python loop, 16 syncs); NTSC 3.39 ms (16 %, 7
`.item()` syncs); NLMeans 2.60 ms (11 %, GPU); rainbow median 0.61 ms but mean 2.0 ms (cuDNN re-autotune on a
random filter length); HF noise 1.13 ms (8 syncs); dithering CPU round trip 0.99 ms (FS) / 1.28 ms (riemersma).
Mixed codecs: HEVC 60 ms per call, WebP 9.6 ms per call (CPU stages = 40 % of the batch).

Found on the way: `jpeg_range` reaching 100 makes DiffJPEG divide by zero (NaN, later a CUDA assert).

### Training pressure (owner's DUAL degradation-test config: batch 16, LR 128, bf16, channels-last, compile)

| compile mode | degradations | feed_data isolated / in loop | train step | ratio |
|---|---|---|---|---|
| max-autotune | all | 199.8 / 203.2 ms | 90.3 ms | 2.2 |
| max-autotune | none | 12.4 / 10.7 ms | 89.9 ms | 0.12 |
| default | all | 198.9 / 238.8 ms | 104.6 ms | 1.9–2.3 |

With everything on, degradation is two-thirds of an iteration. Dithering alone is 158.6 ms (68 % of feed_data):
the owner's `dithering_palette_mode: 0.7` sends 70 % of batches through the CPU palette path (median cut of
128–256 colours, serial over 16 images). `resize_pt` has a 0.26 ms median but an 18 ms mean (p90 74 ms) even with
degradations off: the lanczos kernel shape follows the random scale and cuDNN re-autotunes. Poisson noise 10 ms
(computed then multiplied by 0), DiffJPEG 6.7 ms, NLMeans 5.0 ms, NTSC 4.5 ms. A busy GPU does not slow the
degradation itself (in-loop feed_data equals the isolated number); the cost is simply additive. Cold max-autotune
compile of the DUAL model is about 15 min; warm cache 35 s. Peak training allocation 9.4 GiB.

## Changes

### CUDA dithering kernel (`traiNNer/csrc/dither_kernel.cu`, `dither.cpp`, `traiNNer/data/dither_cuda.py`)
Exact ports of chainner_ext's error diffusion (8 algorithms, anti-diagonal wavefront per image-channel, errors
summed in the serial scan's order) and riemersma (one thread per image-channel walking the reference's Hilbert
traversal). Bit-identical to chainner_ext in 126 test cases (`tests/test_dither_cuda.py`), including the sign of
zero and odd sizes. Wired into `_apply_dithering` (uniform mode) when CUDA is available; riemersma uses the kernel
from batch 2 upward (one Hilbert walk is latency-bound: 3x faster at batch 8, 0.6x at batch 1). Cold build ~40 s,
cached load 0.5 s.

| case | kernel | CPU path | speedup |
|---|---|---|---|
| floydsteinberg b8 128² | 0.22 ms | 1.81 ms | 8.2x |
| stucki b8 256² | 0.67 ms | 8.25 ms | 12.4x |
| riemersma b8 128² | 0.79 ms | 2.37 ms | 3.0x |
| floydsteinberg b1 2048² | 8.3 ms | 114 ms | 13.8x |
| riemersma b1 2048² | 241 ms | 142 ms | 0.6x (routed to CPU) |

### Sync-free noise and DiffJPEG (`traiNNer/data/degradations.py`, `traiNNer/utils/diffjpeg.py`)
Poisson: the per-image `torch.unique` loop became a batched device-side level count (sort, count changes, log2/ceil
in float64); the gray decision is evaluated once; with scale range [0, 0] and `gray_prob` ≤ 0 the noise is skipped
and `clamp(img + 0)` returned. DiffJPEG: `quality_to_factor` takes the whole per-image quality tensor, so the
per-image loop and its 8 syncs per call are gone; the factor has a floor of 1e-7 so quality 100 is finite and
near-lossless (only DiffJPEG's fixed 4:2:0 chroma remains). Bit-identical to the previous code in 116 tests
(`tests/test_otf_sync_free.py`); the only behaviour changes are the [0, 0] skip (no random draws, so a seeded
stream differs after it) and quality ≥ 100 (was NaN).

| case (batch 8×3×256²) | syncs before → after | ms before → after |
|---|---|---|
| Poisson, range [0, 0] (template) | 11 → 0 | 2.54 → 0.02 |
| Poisson, range [0.05, 3], gray 0.4 | 20 → 1 | 4.77 → 1.54 |
| Gaussian, sigma [1, 30] | 2 → 0 | 0.27 → 0.09 |
| DiffJPEG, random quality 30–95 | 8 → 0 | 1.73 → 0.70 |

### Sync-free GPU degradations (`traiNNer/data/gpu_degradations.py`)
Every recurring host sync removed (constants, colour matrices and their inverses, Bayer maps, frequency grids and
NTSC carriers are cached per device/size; the first call of each still syncs once). Rainbow's cuDNN re-autotune
on every new filter length (28–85 ms spikes) is fixed by zero-padding the kernel to a radius of 12, 24 or 48,
bit-identical output. 33 tests compare each function with commit 0ed482ff under `set_sync_debug_mode("error")`.

| function (batch 8×3×256²) | syncs/call | median ms before → after |
|---|---|---|
| ntsc broadcast / vhs_sp | 9 → 0 / 14 → 0 | 2.65 → 1.33 / 4.44 → 2.11 |
| rainbow (mean, incl. spikes) | 0 | 1.05 → 0.37 (max 32.8 → 0.71) |
| channel_shift yuv / chroma_subsample | 3 → 0 / 3 → 0 | 0.44 → 0.10 / 0.63 → 0.27 |
| ordered_dither / lowpass / overshoot | 1 → 0 / 0 / 0 | 0.32 → 0.04 / 0.17 → 0.11 / 0.21 → 0.14 |

**Deviation, kept on purpose:** HEAD keyed NTSC's filter cache on `round(kernel.sum() * 1e8)`, and every kernel
sums to 1, so filters with the same tap count collided: `vhs_sp` ran the 500 kHz colour-under filter in place of
the 3 MHz luma filter (max diff 0.45 at 8×256²), inputs above 754×480 shared one filter for Y, I and Q (max diff
0.72 at 512²), and random VHS bandwidths were ignored. The cache is now keyed on the filter's design parameters;
those paths now give what HEAD's code intended (HEAD with its cache disabled). The default broadcast preset at
256² is bit-identical. `channel_shift_pt` used numpy's unseeded global RNG, so channel shifts were not reproducible
from the run seed; it now draws from the project RNG (same distribution). Because that RNG is shared, a seeded run's
random stream differs from before from the first channel shift onward. Also fixed on the per-image CPU paths
(`otf_degradations.py`): 1-channel images crashed uniform and palette dithering and WebP, and 4-channel palette
dithering misaligned the palette; both now round-trip (alpha passes through palette dithering unchanged).

### Codec path settings (`traiNNer/data/compress_video_batch.py`)
Backends unchanged (owner's benchmarks). What actually reached the encoders was checked by effect on the output:
x265 ran a 32-thread pool with one frame thread (now a pool and frame threads derived from the core count and the
batch size); libvpx started 32 threads (now 1 up to 256², else min(4, cores)); mpeg2's `qscale:v` was not an
encoder option (removed, bitstream identical) and its 16 slice threads became 1; the PyAV decode built a new
colour converter per frame (now one per batch); the `/255` conversion runs on the GPU after a uint8 upload,
dividing by a device tensor so the result matches the CPU bit for bit. h264, hevc, mpeg2 and vp9 decode to exactly
the same tensors as before. 45 tests (`tests/test_compress_batch.py`) check shapes, ranges, odd sizes, that every
frame is an I-frame, and that quality takes effect.

| batch 8, ms before → after | h264 | hevc | mpeg4 | mpeg2 | vp9 |
|---|---|---|---|---|---|
| 128² | 5.6 → 5.1 | 25.7 → 14.8 | 3.2 → 2.8 | 26.0 → 6.8 | 64.1 → 14.5 |
| 256² | 10.6 → 10.1 | 46.6 → 26.6 | 7.5 → 5.8 | 34.4 → 9.8 | 77.4 → 27.1 |
| 512² | 24.4 → 22.5 | 88.0 → 56.6 | 21.0 → 14.8 | 38.8 → 17.9 | 106.7 → 47.8 |

**Deviation, kept on purpose:** mpeg4 silently ignored its quality (`crf` is not an mpeg4 option), so every mpeg4
batch got about q=2 whatever `compress_mpeg4_range` said; it now uses `qmin=qmax=quality` like mpeg2, so the
configured 2–20 range applies (PSNR 35.7 / 30.0 / 26.9 dB at q = 2 / 10 / 20, was 35.4 dB at every q). Training
data with mpeg4 gets the configured artefact strength from now on. Also corrected in the docstring: in CRF mode
x264/x265 set each frame's QP from the running average of earlier frames in the batch, so batch output is
intra-only but not identical to per-image encoding. WebP: `method=0` and the 8-worker pool are correct; Pillow's
WebP decode only partly releases the GIL (1.8x on 8 threads), a backend change was not made.

## After (same script, same arguments; commit b6812173 = kernel + S1 + S2 + S3)

| run (batch 8, GT 256) | feed_data median / p90 before | after | mean before → after |
|---|---|---|---|
| Floyd-Steinberg, JPEG | 21.2 / 24.5 ms | 9.3 / 13.0 ms | 22.5 → 10.0 |
| Riemersma, JPEG | 21.5 / 25.4 ms | 8.6 / 13.4 ms | 23.0 → 9.6 |
| Floyd-Steinberg, mixed codecs | 31.1 / 68.7 ms | 21.5 / 38.3 ms | 42.9 → 25.0 |

The owner's DUAL training-pressure after-run (palette dithering, resize autotune) is recorded below once those
lanes land.

## Decisions
- GPU stays the home of the degradations: their cost is additive per step either way, and hiding them on the CPU
  only works when the user's cores out-run their GPU, which fails on most machines with a big GPU. The measured
  ratio degradation / training step (owner's DUAL degradation-test config) is recorded in the profile file.
- Riemersma at batch 1 goes to the CPU path (identical bits, faster).
- Palette-mode dithering (70 % of events in the owner's config) stays on the CPU this phase; a GPU median cut plus
  palette diffusion kernel is phase 2 if the after-profile shows it matters.

## Re-run
```
cd /home/rift/workspace/scratch/trainner-cuda-dither
export CUDA_HOME=/usr/local/cuda-13.2 PYTHONPATH=$PWD
PY=/home/rift/workspace/traiNNer-redux/venv/bin/python
$PY benchmarks/otf_profile.py --dither floydsteinberg --codecs jpeg  --out benchmarks/otf_profile_after.txt
$PY benchmarks/otf_profile.py --dither riemersma      --codecs jpeg  --out benchmarks/otf_profile_after.txt --append
$PY benchmarks/otf_profile.py --dither floydsteinberg --codecs mixed --out benchmarks/otf_profile_after.txt --append
$PY -m pytest -q tests/test_dither_cuda.py tests/test_otf_sync_free.py tests/test_gpu_degradations_sync_free.py
```
