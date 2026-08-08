"""Package initialization and native runtime load-order guarantees."""

import os

# Inductor's Linux default lives under /tmp, so stopping WSL or clearing its
# temporary directory turns an ordinary training resume into a cold compile.
# Keep compiler artifacts durable by default while preserving explicit user
# overrides and a single root override for portable installations.
if os.name == "posix":
    _compile_cache_root = os.environ.get(
        "TRAINNER_COMPILE_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "traiNNer-redux"),
    )
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(_compile_cache_root, "torchinductor"),
    )
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        os.path.join(_compile_cache_root, "triton"),
    )
    os.environ.setdefault("TRITON_STORE_BINARY_ONLY", "1")

# TensorBoard may eagerly import TensorFlow. With Triton 3.8, importing
# TensorFlow first can crash while libtriton initializes. Preload Triton so
# every traiNNer entry point has deterministic native runtime ordering. Keep
# Triton optional for installations that do not use torch.compile.
if os.name == "posix":
    try:
        import triton as _triton  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "triton":
            raise
