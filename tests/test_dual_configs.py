"""Validate the standalone DUAL family recipes without allocating weights."""
from pathlib import Path

import pytest
import torch

from traiNNer.archs import build_network
from traiNNer.utils.options import yaml_load

ROOT = Path(__file__).resolve().parents[1]
PRESETS = [
    ("light", "LIGHT_C128G4D16_PU2", 128, (4, 3, 3, 6), True),
    ("xs", "XS_C128G4D16", 128, (4, 3, 3, 6), False),
    ("s", "S_C160G6", 160, (6,) * 6, False),
    ("m", "M_C192G6", 192, (6,) * 6, False),
    ("l", "L_C224G10", 224, (6,) * 10, False),
    ("xl", "XL_C256G12", 256, (6,) * 12, False),
]


@pytest.mark.parametrize("factory,suffix,width,depths,unshuffle", PRESETS)
def test_config(factory, suffix, width, depths, unshuffle):
    path = ROOT / "options/train/DUAL" / f"4x_DUAL_{suffix}_pretrain64.yml"
    options, text = yaml_load(str(path))
    assert options.name == path.stem
    # Preserve the user's latest Light recipe edit instead of restoring the
    # original strict settings during a source-to-checkout sync.
    if factory == "light":
        assert options.manual_seed is None
        assert options.deterministic is False
    else:
        assert options.manual_seed == 1024
        assert options.deterministic is True
        assert options.deterministic_warn_only is False
    assert options.scale == options.network_g["scale"] == 4
    assert options.network_g["type"] == f"dual_{factory}"
    assert options.network_g["use_checkpoint"] is False
    assert options.network_g["dual_core_checkpoint"] is False
    assert options.use_amp and options.amp_bf16
    assert options.use_compile and options.compile_mode == "max-autotune"
    assert options.path.pretrain_network_g is None
    assert options.path.resume_state is None
    for key in ("manual_seed:", "deterministic:", "deterministic_warn_only:"):
        assert sum(line.startswith(key) for line in text.splitlines()) == (
            0 if factory == "light" else 1
        )
    with torch.device("meta"):
        model = build_network(options.network_g)
    assert model.embed_dim == width
    assert model.depths == depths
    assert model.num_heads == (width // 32,) * len(depths)
    assert model.unshuffle is unshuffle
    assert len(model.dynamic_conv_names) == len(depths)
    assert all(p.device.type == "meta" for p in model.parameters())
    assert all(
        not m.core_checkpoint for m in model.modules() if hasattr(m, "core_checkpoint")
    )
