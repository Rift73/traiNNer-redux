#!/usr/bin/env python3
"""Compatibility CLI for the reusable DRFT TensorRT graph rewrites."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traiNNer.archs.drft_tensorrt_rewrite import main  # noqa: E402

if __name__ == "__main__":
    main()
