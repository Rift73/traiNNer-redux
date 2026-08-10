from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import torch


def relative_index(query_window: int, key_window: int, device: torch.device) -> torch.Tensor:
    qh = torch.arange(query_window, device=device)
    qw = torch.arange(query_window, device=device)
    kh = torch.arange(key_window, device=device)
    kw = torch.arange(key_window, device=device)
    query = torch.stack(torch.meshgrid(qh, qw, indexing="ij")).flatten(1)
    key = torch.stack(torch.meshgrid(kh, kw, indexing="ij")).flatten(1)
    relative = key[:, None, :] - query[:, :, None]
    relative = relative.permute(1, 2, 0).contiguous()
    relative[:, :, 0] += query_window - 1
    relative[:, :, 1] += query_window - 1
    relative[:, :, 0] *= query_window + key_window - 1
    return relative.sum(-1).long()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--tensorrt-bin", type=Path)
    args = parser.parse_args()

    dll = args.dll.resolve()
    if not dll.is_file():
        raise FileNotFoundError(dll)

    if os.name == "nt":
        if args.tensorrt_bin is None:
            parser.error("--tensorrt-bin is required on Windows")
        tensorrt_bin = args.tensorrt_bin.resolve()
        if not tensorrt_bin.is_dir():
            raise FileNotFoundError(tensorrt_bin)
        os.add_dll_directory(str(tensorrt_bin))
        cuda_bin = Path(os.environ["CUDA_PATH"]) / "bin"
        os.add_dll_directory(str(cuda_bin))

    library = ctypes.CDLL(str(dll))
    launch = library.drftLaunchOCABAttention
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int32
    launch_bias = library.drftLaunchOCABRelativeBias
    launch_bias.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    launch_bias.restype = ctypes.c_int32
    launch_bias_flat = library.drftLaunchOCABRelativeBiasFlat
    launch_bias_flat.argtypes = launch_bias.argtypes
    launch_bias_flat.restype = ctypes.c_int32

    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    batch = 1
    heads = 2
    query_window = 32
    key_window = 40
    query_tokens = query_window**2
    key_tokens = key_window**2
    head_dim = 32
    span = query_window + key_window - 1
    index = relative_index(query_window, key_window, device)

    for bias_heads in (heads, 6):
        bias_table = torch.randn(
            span * span, bias_heads, device=device
        ).to(torch.bfloat16)
        expanded_bias = torch.empty(
            1,
            bias_heads,
            query_tokens,
            key_tokens,
            device=device,
            dtype=torch.bfloat16,
        )
        bias_reference = bias_table[index.reshape(-1)].view(
            query_tokens, key_tokens, bias_heads
        ).permute(2, 0, 1).unsqueeze(0)
        for kernel_name, kernel in (
            ("flat", launch_bias_flat),
            ("q32_k40", launch_bias),
        ):
            bias_error = kernel(
                bias_table.data_ptr(),
                expanded_bias.data_ptr(),
                bias_heads,
                query_window,
                key_window,
                torch.cuda.current_stream().cuda_stream,
            )
            if bias_error != 0:
                raise RuntimeError(
                    f"CUDA relative-bias {kernel_name} launch failed with error {bias_error}"
                )
            torch.cuda.synchronize()
            if not torch.equal(expanded_bias, bias_reference):
                mismatch = int(torch.count_nonzero(expanded_bias != bias_reference))
                raise RuntimeError(
                    f"OCAB relative-bias {kernel_name} expansion is not bit exact: "
                    f"{mismatch}"
                )

        timings = {}
        warmup = 100
        iterations = 1000
        for kernel_name, kernel in (
            ("flat", launch_bias_flat),
            ("q32_k40", launch_bias),
        ):
            for _ in range(warmup):
                error = kernel(
                    bias_table.data_ptr(), expanded_bias.data_ptr(), bias_heads,
                    query_window, key_window,
                    torch.cuda.current_stream().cuda_stream,
                )
                if error != 0:
                    raise RuntimeError(
                        f"CUDA relative-bias {kernel_name} warmup failed with error {error}"
                    )
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                error = kernel(
                    bias_table.data_ptr(), expanded_bias.data_ptr(), bias_heads,
                    query_window, key_window,
                    torch.cuda.current_stream().cuda_stream,
                )
                if error != 0:
                    raise RuntimeError(
                        f"CUDA relative-bias {kernel_name} timing failed with error {error}"
                    )
            end.record()
            end.synchronize()
            timings[kernel_name] = start.elapsed_time(end) / iterations
        print(
            {
                "relative_bias": "passed",
                "heads": bias_heads,
                "elements": expanded_bias.numel(),
                "exact": True,
                "milliseconds": timings,
                "q32_k40_over_flat": timings["q32_k40"] / timings["flat"],
            },
            flush=True,
        )

    reports = []
    for magnitude in (0.125, 1.0):
        query = (
            torch.randn(batch, heads, query_tokens, head_dim, device=device)
            * magnitude
        ).to(torch.bfloat16)
        key_transposed = (
            torch.randn(batch, heads, head_dim, key_tokens, device=device)
            * magnitude
        ).to(torch.bfloat16)
        value = torch.randn(
            batch, heads, key_tokens, head_dim, device=device
        ).to(torch.bfloat16)
        table = (
            torch.randn(span * span, heads, device=device) * 0.02
        ).to(torch.bfloat16)
        output = torch.empty_like(query)
        stream = torch.cuda.current_stream().cuda_stream
        error = launch(
            query.data_ptr(),
            key_transposed.data_ptr(),
            value.data_ptr(),
            table.data_ptr(),
            output.data_ptr(),
            batch,
            heads,
            query_window,
            key_window,
            stream,
        )
        if error != 0:
            raise RuntimeError(f"CUDA launch failed with error {error}")
        torch.cuda.synchronize()

        bias = table.float()[index.reshape(-1)]
        bias = bias.view(query_tokens, key_tokens, heads).permute(2, 0, 1)
        scores = torch.matmul(query.float(), key_transposed.float()) + bias
        reference = torch.matmul(torch.softmax(scores, dim=-1), value.float())
        reference_bf16 = reference.to(torch.bfloat16)
        difference = output.float() - reference_bf16.float()
        relative_l2 = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference_bf16.float()).clamp_min(1e-12)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                output.float().reshape(1, -1),
                reference_bf16.float().reshape(1, -1),
            )
        )
        report = {
            "magnitude": magnitude,
            "finite": bool(torch.isfinite(output).all()),
            "max_abs": float(difference.abs().max()),
            "mean_abs": float(difference.abs().mean()),
            "relative_l2": relative_l2,
            "cosine": cosine,
        }
        reports.append(report)
        print(report, flush=True)
        if not report["finite"] or relative_l2 > 0.01 or cosine < 0.9999:
            raise RuntimeError(f"OCAB kernel numerical admission failed: {report}")

    print({"status": "passed", "cases": reports}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
