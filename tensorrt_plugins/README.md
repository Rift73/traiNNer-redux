# DRFT TensorRT plugins

This directory contains DRFT's TensorRT deployment plugins. The normal DRFT
ONNX remains portable; `rewrite_onnx_for_tensorrt_plugins` creates a separate
TensorRT-specialized ONNX and refuses to overwrite either artifact.

The admitted optimized path keeps TensorRT's native fused MHA and applies four
rewrites: exact compact Q32/K48 relative bias, blockwise dense-fusion MatMuls,
exact image-layer-normalization plugins, and arithmetic-free token-major OCAB
Q/K/V packing. TensorRT still owns QK, softmax, PV, projections, convolutions,
SwiGLU, channel attention, and reconstruction. The accepted plugin library is
`DrftTensorRTPluginsOptimizedV8`.

`compact_bias` remains the portable-plugin fallback. The experimental
`fused_attention` strategy is negative evidence: replacing native MHA reduced
storage but was 26.8% slower at Micro p50.

## Export and rewrite

Export DRFT with a `1x3x96x96` example and dynamic H/W, then create a distinct
optimized ONNX:

```python
from drft_arch import rewrite_onnx_for_tensorrt_plugins

report = rewrite_onnx_for_tensorrt_plugins(
    "drft.portable.onnx",
    "drft.tensorrt_optimized.onnx",
)
```

The default strategy is `optimized`. It verifies every compact bias, preserves
the native MHA patterns, and reports the exact number of all plugin nodes. The
accepted graph contract is batch 1 with H/W divisible by 32. The engine profile
is static and may use a different H/W, such as `1x3x512x512`.

## Build the plugin on Windows

```powershell
cmake -S . -B build_trt112_windows -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" `
  -DTENSORRT_ROOT=C:/path/to/TensorRT-11.2.1.2
cmake --build build_trt112_windows --config Release --parallel
```

Build the accepted target explicitly:

```powershell
cmake --build build_trt112_windows --config Release --parallel `
  --target DrftTensorRTPluginsOptimizedV8
```

The DLL is written to
`build_trt112_windows/Release/DrftTensorRTPluginsOptimizedV8.dll`.

## Build the plugin on Linux or WSL

Install the TensorRT development packages and CUDA toolkit first. For the
Ubuntu packages, use `/usr` as `TENSORRT_ROOT`; the build locates the
multiarch include and library directories beneath it.

```bash
cmake -S . -B build_trt112_linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=/usr \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_trt112_linux --parallel \
  --target DrftTensorRTPluginsOptimizedV8
```

The shared library is written to
`build_trt112_linux/libDrftTensorRTPluginsOptimizedV8.so`.

The default build contains native images for SM 80, 86, 89, 90, and 120 plus
PTX for SM 120. Change `DRFT_CUDA_ARCHITECTURES` when a different deployment
fleet is required. TensorRT engines remain specific to the TensorRT/plugin
version, build flags, profile, and target GPU; rebuild the engine on an
incompatible target rather than copying an admitted engine blindly.

## Build one self-contained engine

```powershell
trtexec.exe `
  --onnx=drft.tensorrt_optimized.onnx `
  --saveEngine=drft.engine `
  --skipInference `
  --builderOptimizationLevel=3 `
  --memPoolSize=workspace:8G `
  --maxAuxStreams=0 `
  --profilingVerbosity=detailed `
  --noTF32 `
  --dynamicPlugins=DrftTensorRTPluginsOptimizedV8.dll `
  --setPluginsToSerialize=DrftTensorRTPluginsOptimizedV8.dll
```

`--setPluginsToSerialize` embeds the plugin library in the engine. Some
TensorRT releases can log duplicate-creator errors during their own post-save
self-reload when the library is both dynamic and serialized, even after
reporting a successful build. Accept the saved artifact only when `trtexec`
reached `PASSED` and a fresh process loads and validates the engine without
the external library. Do not rebuild merely to remove that downstream
diagnostic. Build each exact engine identity once and reuse it for validation
and timing.

`libDrftTensorRTPluginsV3` / `DrftTensorRTPluginsV3.dll` remains the original
Q32 / centered Halo40 plugin and is intentionally unchanged. DRFT Safe uses a
separate `DrftTensorRTPluginsSafeV1` library with the distinct
`DrftOCABRelativeBiasSafe_TRT` custom op. Its constant-specialized compact-bias
kernel admits Q32/K48 as well as Q32/K40, while the original library continues
to reject anything except Q32/K40. Build only the Safe library with:

```bash
cmake --build build_trt112_linux --target DrftTensorRTPluginsSafe --parallel
```

Both compact-bias plugins require BF16 bias tensors. The portable ONNX remains
the fallback for other geometry or runtimes.

## Precision and validation

The blockwise dense-fusion rewrite is algebraically exact, but it moves BF16
rounding sites and therefore is not promised to be bit-exact. The image-LN
plugin reproduces the exported BF16 rounding chain, and the Q/K/V pack plugin
performs only byte-preserving data movement. Validate the built engine against
the original PyTorch model on representative trained-checkpoint images before
making a quality claim. TensorRT may also select FP16 execution for some BF16
depthwise convolutions; inspect the final engine layer dump rather than inferring
precision solely from the ONNX dtypes.

See `DRFT_TENSORRT_PLUGIN_RESULTS_V1.json` for the original compact-bias result.
The accepted DRFT-Light V8 campaign is recorded in
`results/benchmarks/drft_light_trt_lossless_optimization_v1/DRFT_LIGHT_TENSORRT_OPTIMIZATION_RESULTS_V1.json`
inside the traiNNer-redux workspace.
