"""Durable, architecture-safe runtime support for ``torch.compile`` caches."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import threading
from pathlib import Path
from typing import Any

import torch

_INSTALL_LOCK = threading.Lock()
_COMPILE_CACHE_RUNTIME_ABI = "trainner-compile-cache-v2"
_PATCH_MARKER = "_trainner_stable_autotune_identity_v2"
_GUARD_GRAD_PATCH_MARKER = "_trainner_missing_guard_grad_v1"


@functools.cache
def _runtime_fingerprint() -> str:
    try:
        import triton

        triton_version: str | None = triton.__version__
    except (ImportError, AttributeError):
        triton_version = None

    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "major": properties.major,
            "minor": properties.minor,
            "total_memory": properties.total_memory,
        }

    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
        "gpu": gpu,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def install_stable_triton_autotune_identity() -> None:
    """Keep bundled static autotuners reusable across fresh processes.

    PyTorch 2.14 serializes ``CachingAutotuner.filename`` as ``None`` inside
    Triton bundles. Its best-config cache is filename-keyed, so every restored
    package otherwise benchmarks every candidate again. The bundle cache key is
    already a stable hash of the full generated Triton source and torch ABI; use
    it as a synthetic filename before both the live and serialized autotuners
    are created.
    """
    from torch._inductor.triton_bundler import TritonBundler

    current = TritonBundler.put_static_autotuner
    if getattr(current, _PATCH_MARKER, False):
        return

    with _INSTALL_LOCK:
        current = TritonBundler.put_static_autotuner
        if getattr(current, _PATCH_MARKER, False):
            return

        original = current.__func__

        def stable_put_static_autotuner(
            cls: type[TritonBundler],
            key: str,
            kernel: Any,
        ) -> None:
            from torch._inductor.runtime.cache_dir_utils import cache_dir
            from torch._inductor.runtime.triton_heuristics import (
                check_autotune_cache,
                triton_config_to_hashable,
            )

            identity_dir = Path(cache_dir()) / "stable_autotune_identity_v2"
            identity_dir.mkdir(parents=True, exist_ok=True)
            kernel.filename = str(
                identity_dir / f"{key}-{_runtime_fingerprint()}.py"
            )

            configs = [result.config for result in kernel.compile_results]
            cached_configs, autotune_cache, cache_info = check_autotune_cache(
                configs,
                kernel.filename,
                kernel.inductor_meta,
                dynamic_scale_rblock_eligible=kernel._could_rblock_scale,  # noqa: SLF001
            )
            kernel.autotune_cache_info = cache_info
            kernel.save_cache_hook = (
                autotune_cache.save if autotune_cache is not None else None
            )

            if len(cached_configs) == 1 and len(configs) > 1:
                selected = triton_config_to_hashable(cached_configs[0])
                matches = [
                    result
                    for result in kernel.compile_results
                    if triton_config_to_hashable(result.config) == selected
                ]
                if len(matches) == 1:
                    kernel.compile_results = matches
            original(cls, key, kernel)

        setattr(stable_put_static_autotuner, _PATCH_MARKER, True)
        TritonBundler.put_static_autotuner = classmethod(
            stable_put_static_autotuner
        )


def install_dynamo_guard_grad_compatibility() -> None:
    """Restore cached guard tensors whose unguarded grad was omitted.

    PyTorch 2.14's experimental precompile serializer replaces a tensor value
    outside the guard tree with ``_Missing``. A model parameter can have a
    populated ``.grad`` when the package is saved, so that gradient is encoded
    as ``_Missing`` and the stock loader later assigns the sentinel directly to
    ``Tensor.grad``. Treating an omitted, unguarded gradient as ``None`` matches
    a freshly constructed parameter and keeps the otherwise-valid package
    reusable.
    """
    from torch._dynamo import guards as dynamo_guards

    pickler = dynamo_guards.GuardsStatePickler
    current = pickler._unpickle_tensor  # noqa: SLF001
    current_function = getattr(current, "__func__", current)
    if getattr(current_function, _GUARD_GRAD_PATCH_MARKER, False):
        return

    with _INSTALL_LOCK:
        current = pickler._unpickle_tensor  # noqa: SLF001
        current_function = getattr(current, "__func__", current)
        if getattr(current_function, _GUARD_GRAD_PATCH_MARKER, False):
            return

        original = current_function

        def compatible_unpickle_tensor(
            cls: type[Any],
            meta_tensor: torch.Tensor,
            device: torch.device,
            pytype: type,
            dispatch_keys_raw: int,
            grad: Any,
        ) -> torch.Tensor:
            if isinstance(grad, dynamo_guards._Missing):  # noqa: SLF001
                grad = None
            return original(
                cls,
                meta_tensor,
                device,
                pytype,
                dispatch_keys_raw,
                grad,
            )

        setattr(
            compatible_unpickle_tensor,
            _GUARD_GRAD_PATCH_MARKER,
            True,
        )
        pickler._unpickle_tensor = classmethod(  # noqa: SLF001
            compatible_unpickle_tensor
        )


def scope_dynamo_precompile_cache(cache_key_tag: str) -> str:
    """Give Dynamo's package cache the ABI dimension its source key omits."""
    if not cache_key_tag:
        raise ValueError("compile cache scope requires a non-empty cache key tag")

    from torch._dynamo.package import DynamoCache, cache_dir

    payload = (
        f"{_COMPILE_CACHE_RUNTIME_ABI}|{cache_key_tag}|"
        f"runtime={_runtime_fingerprint()}"
    ).encode()
    scope = hashlib.sha256(payload).hexdigest()
    path = Path(cache_dir()) / "dynamo-scoped-v1" / scope
    path.mkdir(parents=True, exist_ok=True)
    path_string = os.fspath(path)
    # DiskDynamoCache overrides DiskDynamoStore.path_prefix() with a hard-coded
    # global directory, so changing its inherited _path_prefix field is a no-op.
    # Install an instance-local callable instead; PrecompileContext holds this
    # same singleton object.
    DynamoCache.path_prefix = lambda: path_string  # type: ignore[method-assign]
    return path_string


def configure_compile_cache_runtime(cache_key_tag: str) -> str:
    """Install stable kernel identity and select the matching package scope."""
    install_dynamo_guard_grad_compatibility()
    install_stable_triton_autotune_identity()
    return scope_dynamo_precompile_cache(cache_key_tag)
