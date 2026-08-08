from __future__ import annotations

from pathlib import Path
from typing import Any

from traiNNer.utils import compile_cache


def test_dynamo_precompile_cache_is_scoped_by_abi() -> None:
    from torch._dynamo.package import DynamoCache

    first = compile_cache.scope_dynamo_precompile_cache("drft-abi-a")
    repeated = compile_cache.scope_dynamo_precompile_cache("drft-abi-a")
    second = compile_cache.scope_dynamo_precompile_cache("drft-abi-b")

    assert first == repeated
    assert first != second
    assert DynamoCache.path_prefix() == second
    assert Path(second).is_dir()


def test_static_autotuner_gets_stable_filename(monkeypatch: Any) -> None:
    from torch._inductor.triton_bundler import TritonBundler

    captured: list[tuple[str, Any]] = []

    def capture(cls: type[TritonBundler], key: str, kernel: Any) -> None:
        del cls
        captured.append((key, kernel))

    monkeypatch.setattr(
        TritonBundler,
        "put_static_autotuner",
        classmethod(capture),
    )
    compile_cache.install_stable_triton_autotune_identity()

    class FakeKernel:
        def __init__(self) -> None:
            self.compile_results: list[Any] = []
            self.inductor_meta: dict[str, Any] = {}
            self._could_rblock_scale = False
            self.filename: str | None = None
            self.autotune_cache_info: dict[str, Any] | None = None
            self.save_cache_hook: Any = None

    kernel = FakeKernel()
    TritonBundler.put_static_autotuner("stable-source-key", kernel)

    assert captured == [("stable-source-key", kernel)]
    assert kernel.filename is not None
    assert "stable_autotune_identity_v2" in kernel.filename
    assert Path(kernel.filename).name.startswith("stable-source-key-")
