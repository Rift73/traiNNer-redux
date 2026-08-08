# ruff: noqa: SLF001

from unittest.mock import patch

from torch._dynamo.precompile_context import PrecompileContext
from traiNNer.models.base_model import BaseModel


def test_compile_cache_publication_failure_does_not_abort_training() -> None:
    model = object.__new__(BaseModel)
    model._compile_precompile_cache_enabled = True
    model._compile_precompile_cache_signature = (0, 0)

    with (
        patch.object(PrecompileContext, "_dynamo_cache_entries", [object()]),
        patch.object(
            PrecompileContext,
            "_backend_artifacts_by_key",
            {"artifact": object()},
        ),
        patch.object(
            PrecompileContext,
            "save_to_dynamo_cache",
            side_effect=RuntimeError("unserializable compiled loss"),
        ),
    ):
        model.finalize_compile_cache()

    assert model._compile_precompile_cache_enabled is False
    assert model._compile_precompile_cache_signature == (0, 0)
