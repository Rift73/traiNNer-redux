#!/usr/bin/env python3
"""Export the current canonical DRFT-Light TensorRT optimization baseline.

The portable ONNX is traced at 96x96 with dynamic spatial axes.  The reference
and eventual TensorRT engine use one fixed 512x512 input.  This script exports
only; every unique engine is built separately exactly once.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto

REPO_ROOT = Path("/home/rift/workspace/traiNNer-redux")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traiNNer.archs.drft_arch import (  # noqa: E402
    OCAB,
    ImageLayerNorm,
    NarrowWindowAttention,
    WindowAttentionRFB,
    drft_light,
    rewrite_onnx_for_tensorrt_plugins,
)

OUTPUT_ROOT = (
    REPO_ROOT / "results/benchmarks/drft_light_trt_lossless_optimization_v1"
)
EXAMPLE_SHAPE = (1, 3, 96, 96)
ENGINE_SHAPE = (1, 3, 512, 512)
OUTPUT_SHAPE = (1, 3, 2048, 2048)
SEED = 1024
FP8_TYPES = {
    value
    for name in (
        "FLOAT8E4M3FN",
        "FLOAT8E4M3FNUZ",
        "FLOAT8E5M2",
        "FLOAT8E5M2FNUZ",
        "FLOAT8E8M0",
    )
    if (value := getattr(TensorProto, name, None)) is not None
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_param if dim.dim_param else dim.dim_value
        for dim in value_info.type.tensor_type.shape.dim
    ]


def audit_onnx(path: Path) -> dict[str, object]:
    model = onnx.load(str(path), load_external_data=False)
    plugins = Counter(
        node.op_type
        for node in model.graph.node
        if node.op_type.startswith("Drft") and node.op_type.endswith("_TRT")
    )
    if not plugins:
        onnx.checker.check_model(str(path))

    input_shape = tensor_shape(model.graph.input[0])
    output_shape = tensor_shape(model.graph.output[0])
    if input_shape[:2] != [1, 3] or not all(
        isinstance(dim, str) and dim for dim in input_shape[2:]
    ):
        raise RuntimeError(f"Input H/W are not dynamic: {input_shape}")
    if output_shape[1] != 3 or not all(
        isinstance(dim, str) and dim for dim in output_shape[2:]
    ):
        raise RuntimeError(f"Output H/W are not dynamic: {output_shape}")
    if (
        model.graph.input[0].type.tensor_type.elem_type != TensorProto.FLOAT
        or model.graph.output[0].type.tensor_type.elem_type != TensorProto.FLOAT
    ):
        raise RuntimeError("Host I/O must remain FP32")

    initializer_types = Counter(item.data_type for item in model.graph.initializer)
    return {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_types": {
            TensorProto.DataType.Name(dtype): count
            for dtype, count in sorted(initializer_types.items())
        },
        "plugin_nodes": dict(sorted(plugins.items())),
        "fp8_initializers": sum(initializer_types[dtype] for dtype in FP8_TYPES),
    }


def export_dynamic_onnx(
    model: torch.nn.Module,
    sample: torch.Tensor,
    path: Path,
) -> None:
    torch.onnx.export(
        model,
        (sample,),
        str(path),
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "output_height", 3: "output_width"},
        },
        do_constant_folding=True,
    )


def save_reference(
    model: torch.nn.Module,
    input_array: np.ndarray,
    path: Path,
) -> None:
    input_tensor = torch.from_numpy(np.ascontiguousarray(input_array)).cuda()
    with torch.inference_mode():
        output = model(input_tensor)
    if tuple(output.shape) != OUTPUT_SHAPE:
        raise RuntimeError(f"Unexpected reference shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Reference contains non-finite values")
    output.float().cpu().numpy().tofile(path)
    del input_tensor, output


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "export_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")

    portable_path = OUTPUT_ROOT / "drft_light_x4_dynamic96_portable_mixed.onnx"
    baseline_path = OUTPUT_ROOT / "drft_light_x4_dynamic96_compact_k48_mixed.onnx"
    input_path = OUTPUT_ROOT / "input_1x3x512x512_fp32.bin"
    reference_path = OUTPUT_ROOT / "reference_512x512_to_2048x2048_fp32.bin"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    input_array = np.random.default_rng(SEED).random(ENGINE_SHAPE, dtype=np.float32)
    if input_path.exists():
        preserved_input = np.fromfile(input_path, dtype=np.float32).reshape(
            ENGINE_SHAPE
        )
        if not np.array_equal(input_array, preserved_input):
            raise RuntimeError("Partial input does not match the seed-1024 input")
    else:
        input_array.tofile(input_path)
    model = drft_light(
        scale=4,
        use_checkpoint=False,
        drop_path_rate=0.1,
        attn_type="masked",
        rank=32,
    ).eval().cuda().requires_grad_(False)
    architecture = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embed_dim": model.embed_dim,
        "groups": model.num_layers,
        "depths": list(model.depths),
        "heads": list(model.num_heads),
        "overlap_window_size": model.overlap_window_size,
        "image_layer_norm_sites": sum(
            isinstance(module, ImageLayerNorm) for module in model.modules()
        ),
        "narrow_attention_sites": sum(
            isinstance(module, NarrowWindowAttention) for module in model.modules()
        ),
        "full_width_attention_sites": sum(
            isinstance(module, WindowAttentionRFB) for module in model.modules()
        ),
        "ocab_sites": sum(isinstance(module, OCAB) for module in model.modules()),
    }
    expected = {
        "embed_dim": 96,
        "groups": 2,
        "depths": [6, 6],
        "heads": [3, 3],
        "overlap_window_size": 48,
        "ocab_sites": 2,
    }
    for key, value in expected.items():
        if architecture[key] != value:
            raise RuntimeError(
                f"Canonical DRFT-Light mismatch for {key}: "
                f"expected {value}, got {architecture[key]}"
            )

    deployment = model.prepare_for_onnx_export(
        EXAMPLE_SHAPE,
        precision="tensorrt_mixed",
        dynamic_spatial=True,
    ).cuda().eval().requires_grad_(False)
    deployment_inventory = deployment.onnx_deployment_inventory
    # Generate the large reference before any small-shape inference or legacy
    # export.  The current PyTorch development build can leave cuDNN's plan
    # finder unable to select the first full-resolution convolution after the
    # same module has executed the 96x96 trace shape.
    if reference_path.exists():
        expected_bytes = int(np.prod(OUTPUT_SHAPE)) * np.dtype(np.float32).itemsize
        if reference_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Partial reference has the wrong size: {reference_path.stat().st_size}"
            )
    else:
        save_reference(deployment, input_array, reference_path)
    sample = torch.from_numpy(
        np.ascontiguousarray(input_array[:, :, :96, :96])
    ).cuda()
    with torch.inference_mode():
        sample_output = deployment(sample)
    if tuple(sample_output.shape) != (1, 3, 384, 384):
        raise RuntimeError(f"Unexpected trace output: {tuple(sample_output.shape)}")
    if not torch.isfinite(sample_output).all():
        raise RuntimeError("Trace output contains non-finite values")
    if portable_path.exists():
        audit_onnx(portable_path)
    else:
        export_dynamic_onnx(deployment, sample, portable_path)
    del sample, sample_output, deployment, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if baseline_path.exists():
        plugin_report: dict[str, object] | None = None
    else:
        plugin_report = rewrite_onnx_for_tensorrt_plugins(
            portable_path,
            baseline_path,
            strategy="compact_bias",
        )
    portable_audit = audit_onnx(portable_path)
    baseline_audit = audit_onnx(baseline_path)
    initializers = baseline_audit["initializer_types"]
    if (
        initializers.get("BFLOAT16", 0) <= 0
        or initializers.get("FLOAT16", 0) <= 0
        or baseline_audit["fp8_initializers"] != 0
        or baseline_audit["plugin_nodes"]
        != {"DrftOCABRelativeBiasSafe_TRT": 2}
    ):
        raise RuntimeError(f"Baseline precision/plugin audit failed: {baseline_audit}")

    source_path = REPO_ROOT / "traiNNer/archs/drft_arch.py"
    manifest = {
        "status": "exported_and_audited",
        "scope": "speed-only seed-1024 weights; no quality claim",
        "scale": 4,
        "example_shape": list(EXAMPLE_SHAPE),
        "dynamic_spatial_onnx": True,
        "engine_shape": list(ENGINE_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "precision_policy": "FP32 I/O + BF16 body + FP16 reconstruction tail",
        "seed": SEED,
        "architecture": architecture,
        "deployment_inventory": deployment_inventory,
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "input": {"path": str(input_path), "sha256": sha256(input_path)},
        "reference": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
        },
        "portable_onnx": {
            "path": str(portable_path),
            "sha256": sha256(portable_path),
            "audit": portable_audit,
        },
        "baseline_onnx": {
            "path": str(baseline_path),
            "sha256": sha256(baseline_path),
            "strategy": "compact_bias",
            "plugin_report": plugin_report,
            "reused_completed_plugin_rewrite": plugin_report is None,
            "audit": baseline_audit,
        },
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "onnx": onnx.__version__,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
