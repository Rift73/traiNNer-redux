"""CPU-only recovery tests, including real subprocess fault injection."""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from traiNNer import training_recovery as recovery_runtime
from traiNNer.training_recovery import (
    NONFINITE_EXIT_CODE,
    REPORT_ENV,
    RESUME_ENV,
    SEED_ENV,
    CheckpointLayout,
    CheckpointRetention,
    PersistentNonfiniteError,
    RecoveryBudget,
    RecoveryCheckpoint,
    TrainingRecovery,
    maybe_supervise_training,
    supervise,
)


@pytest.fixture(autouse=True)
def isolated_recovery_environment(monkeypatch: Any) -> None:
    for name in (
        REPORT_ENV,
        RESUME_ENV,
        SEED_ENV,
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "SLURM_PROCID",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("extra", "max_restarts", "delay"),
    [
        ([], 2, 5.0),
        (["--recovery-max-restarts", "4", "--recovery-delay", "0.25"], 4, 0.25),
    ],
)
def test_normal_command_is_supervised_with_original_options(
    tmp_path: Path, monkeypatch: Any, extra: list[str], max_restarts: int, delay: float
) -> None:
    supervisor = Mock(return_value=7)
    monkeypatch.setattr(recovery_runtime, "supervise", supervisor)
    script = tmp_path / "train.py"
    args = [
        "-opt",
        "options/train/DRFT/4x_DRFT_XL.yml",
        "--auto_resume",
        "--name",
        "experiment with spaces",
        "--launcher",
        "none",
        *extra,
    ]
    with pytest.raises(SystemExit) as stopped:
        maybe_supervise_training(str(script), args)
    assert stopped.value.code == 7
    supervisor.assert_called_once_with(
        [sys.executable, "-u", str(script), *args],
        max_restarts=max_restarts,
        restart_delay=delay,
    )


def test_supervised_worker_never_launches_a_second_supervisor(monkeypatch: Any) -> None:
    supervisor = Mock()
    monkeypatch.setattr(recovery_runtime, "supervise", supervisor)
    monkeypatch.setenv(REPORT_ENV, "/tmp/owned-failure-report.json")
    maybe_supervise_training("train.py", ["-opt", "unused.yml"])
    supervisor.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--help"],
        ["-opt", "unused.yml", "-h"],
        ["-opt", "unused.yml", "--no-auto-recover"],
    ],
)
def test_help_missing_config_and_opt_out_do_not_launch_workers(
    monkeypatch: Any, args: list[str]
) -> None:
    supervisor = Mock()
    monkeypatch.setattr(recovery_runtime, "supervise", supervisor)
    maybe_supervise_training("train.py", args)
    supervisor.assert_not_called()


@pytest.mark.parametrize(
    ("env", "args"),
    [
        ({"WORLD_SIZE": "2"}, []),
        ({"WORLD_SIZE": "1", "RANK": "0"}, []),
        ({"LOCAL_RANK": "0"}, []),
        ({"SLURM_PROCID": "0"}, []),
        ({}, ["--launcher", "pytorch"]),
        ({}, ["--launcher=slurm"]),
    ],
)
def test_distributed_launches_keep_ordinary_training_without_rank_restarts(
    monkeypatch: Any, capsys: Any, env: dict[str, str], args: list[str]
) -> None:
    supervisor = Mock()
    monkeypatch.setattr(recovery_runtime, "supervise", supervisor)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    maybe_supervise_training("train.py", ["-opt", "unused.yml", *args])
    supervisor.assert_not_called()
    assert "disabled for distributed launchers" in capsys.readouterr().out


def test_native_windows_keeps_ordinary_training(monkeypatch: Any, capsys: Any) -> None:
    supervisor = Mock()
    monkeypatch.setattr(recovery_runtime, "supervise", supervisor)
    monkeypatch.setattr(
        recovery_runtime, "os", SimpleNamespace(name="nt", environ=os.environ)
    )
    maybe_supervise_training("train.py", ["-opt", "unused.yml"])
    supervisor.assert_not_called()
    assert "requires native WSL/Linux" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["--recovery-max-restarts", "0"],
        ["--recovery-delay", "-1"],
        ["--recovery-delay", "nan"],
        ["--recovery-delay", "inf"],
    ],
)
def test_invalid_recovery_options_fail_before_launch(
    monkeypatch: Any, args: list[str]
) -> None:
    supervisor = Mock()
    monkeypatch.setattr(recovery_runtime, "supervise", supervisor)
    with pytest.raises(SystemExit) as stopped:
        maybe_supervise_training("train.py", ["-opt", "unused.yml", *args])
    assert stopped.value.code == 2
    supervisor.assert_not_called()


def _entrypoint_bootstrap() -> str:
    """Use the real train.py bootstrap, stopping before any training imports."""
    entrypoint = Path(__file__).resolve().parents[2] / "train.py"
    tree = ast.parse(entrypoint.read_text())
    stop = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.ImportFrom) and node.module == "torch.distributed"
    )
    return ast.unparse(ast.Module(body=tree.body[:stop], type_ignores=[]))


def test_normal_entrypoint_waits_without_importing_torch() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (
        "import sys\n"
        "from traiNNer import training_recovery as recovery\n"
        "def supervise(command, **kwargs):\n"
        "    assert 'torch' not in sys.modules\n"
        "    print('supervised-before-torch')\n"
        "    return 0\n"
        "recovery.supervise = supervise\n"
        "sys.argv = ['train.py', '-opt', 'unused.yml', '--auto_resume']\n"
        f"exec({_entrypoint_bootstrap()!r}, "
        f"{{'__name__': '__main__', '__file__': {str(root / 'train.py')!r}}})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "supervised-before-torch" in result.stdout


def test_imported_train_entrypoint_does_not_start_a_supervisor(
    monkeypatch: Any,
) -> None:
    supervisor = Mock()
    monkeypatch.setattr(recovery_runtime, "supervise", supervisor)
    exec(_entrypoint_bootstrap(), {"__name__": "train", "__file__": "train.py"})
    supervisor.assert_not_called()


@pytest.mark.parametrize("is_train", [True, False])
def test_recovery_controls_are_in_normal_training_help_only(
    monkeypatch: Any, capsys: Any, is_train: bool
) -> None:
    from traiNNer.utils.options import parse_options

    monkeypatch.setattr(sys, "argv", ["train.py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        parse_options("unused-root", is_train=is_train)
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--no-auto-recover", "--recovery-max-restarts", "--recovery-delay"):
        assert (flag in help_text) == is_train


def test_train_pipeline_help_exits_before_cuda_checks(monkeypatch: Any) -> None:
    import train as entry

    monkeypatch.setattr(entry.Config, "_config", None)
    monkeypatch.setattr(entry.Config, "_args", None)
    monkeypatch.setattr(sys, "argv", ["train.py", "--help"])
    cuda_check = Mock(side_effect=AssertionError("Help must not check CUDA"))
    monkeypatch.setattr(entry.torch.cuda, "is_available", cuda_check)
    with pytest.raises(SystemExit) as stopped:
        entry.train_pipeline("unused-root")
    assert stopped.value.code == 0
    cuda_check.assert_not_called()


def _archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"test checkpoint")


def _safetensors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)


def _layout(
    root: Path,
    *,
    ema: bool = True,
    discriminator: bool = False,
    extension: str = "safetensors",
) -> CheckpointLayout:
    return CheckpointLayout(
        states=root / "training_states",
        models=root / "models",
        resume_models=root / "models" / "resume_models",
        interval=1000,
        extension=extension,
        ema=ema,
        discriminator=discriminator,
    )


def _checkpoint(layout: CheckpointLayout, iteration: int) -> RecoveryCheckpoint:
    checkpoint = RecoveryCheckpoint(iteration, layout.states / f"{iteration}.state")
    for path in layout.files(checkpoint):
        if path.suffix == ".safetensors":
            _safetensors(path)
        else:
            _archive(path)
    return checkpoint


def test_latest_regular_excludes_manual_partial_and_unpaired_saves(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    good = _checkpoint(layout, 176_000)
    _checkpoint(layout, 176_143)
    _archive(layout.states / "177000.state")
    partial = _checkpoint(layout, 178_000)
    layout.files(partial)[1].write_bytes(b"truncated")
    (layout.states / "999999.state.bak").write_text("backup")
    assert layout.latest_regular() == good


@pytest.mark.parametrize("ema", [True, False])
@pytest.mark.parametrize("discriminator", [True, False])
def test_complete_requires_every_matching_network(
    tmp_path: Path, ema: bool, discriminator: bool
) -> None:
    layout = _layout(tmp_path, ema=ema, discriminator=discriminator)
    checkpoint = _checkpoint(layout, 176_000)
    assert layout.complete(checkpoint)
    layout.files(checkpoint)[-1].unlink()
    assert not layout.complete(checkpoint)


def test_saved_checkpoint_anchor_advances_only_after_complete_fresh_save(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    first = _checkpoint(layout, 176_000)
    recovery = TrainingRecovery(layout, tmp_path / "failure.json")
    recovery.note_resume(str(first.state_path), first.iteration)
    existing = recovery.before_periodic_save(176_000)
    before = recovery.before_periodic_save(177_000)
    assert not recovery.note_periodic_save(176_000, existing)
    assert not recovery.note_periodic_save(177_000, before)
    assert recovery.checkpoint == first
    newer = _checkpoint(layout, 177_000)
    assert recovery.note_periodic_save(177_000, before)
    assert recovery.checkpoint == newer


def test_forced_checkpoint_wins_over_auto_resume(
    tmp_path: Path, monkeypatch: Any
) -> None:
    layout = _layout(tmp_path)
    expected = _checkpoint(layout, 176_000)
    _checkpoint(layout, 177_000)
    recovery = TrainingRecovery(layout, tmp_path / "failure.json")
    monkeypatch.setenv(RESUME_ENV, str(expected.state_path))
    options = SimpleNamespace(auto_resume=True, path=SimpleNamespace(resume_state=None))
    assert recovery.choose_resume(options) == str(expected.state_path)


def test_auto_resume_accepts_the_latest_complete_manual_checkpoint(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    _checkpoint(layout, 186_000)
    manual = _checkpoint(layout, 186_543)
    _archive(layout.states / "187000.state")
    (layout.states / "999999.state.bak").write_text("backup")
    recovery = TrainingRecovery(layout, tmp_path / "failure.json")
    options = SimpleNamespace(auto_resume=True, path=SimpleNamespace(resume_state=None))
    assert recovery.choose_resume(options) == str(manual.state_path)
    recovery.note_resume(str(manual.state_path), manual.iteration)
    assert recovery.checkpoint == manual


def test_auto_resume_will_not_start_over_when_only_incomplete_saves_exist(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    _archive(layout.states / "186543.state")
    recovery = TrainingRecovery(layout, tmp_path / "failure.json")
    options = SimpleNamespace(auto_resume=True, path=SimpleNamespace(resume_state=None))
    with pytest.raises(RuntimeError, match="No complete checkpoint"):
        recovery.choose_resume(options)


def test_explicit_complete_resume_can_be_an_initial_anchor(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    expected = _checkpoint(layout, 176_143)
    recovery = TrainingRecovery(layout, tmp_path / "failure.json")
    recovery.note_resume(str(expected.state_path), expected.iteration)
    assert recovery.checkpoint == expected


def test_failure_report_keeps_good_files_and_never_saves_failed_weights(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    checkpoint = _checkpoint(layout, 176_000)
    recovery = TrainingRecovery(layout, tmp_path / "failure.json")
    recovery.note_resume(str(checkpoint.state_path), checkpoint.iteration)
    before = {path: path.read_bytes() for path in layout.files(checkpoint)}
    failure = PersistentNonfiniteError("nonfinite_generator_loss", 176_143, 20)
    recovery.report_failure(failure, 7970)
    report = json.loads(recovery.report_path.read_text())
    assert report["candidate_iter"] == 176_143
    assert report["consecutive_rejections"] == 20
    assert report["checkpoint"]["iteration"] == 176_000
    assert report["manual_seed"] == 7970
    assert not (layout.states / "176143.state").exists()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_budget_limits_stalled_restarts_but_allows_weeks_of_forward_progress() -> None:
    budget = RecoveryBudget(max_restarts=2)
    assert budget.allow(176_000)
    assert budget.allow(176_000)
    assert not budget.allow(176_000)
    for iteration in range(177_000, 800_000, 1000):
        assert budget.allow(iteration)
        assert budget.restarts == 1


def test_older_checkpoint_does_not_reset_restart_budget() -> None:
    budget = RecoveryBudget(max_restarts=2)
    assert budget.allow(176_000)
    assert budget.allow(175_000)
    assert not budget.allow(174_000)


@pytest.mark.parametrize("ema", [True, False])
@pytest.mark.parametrize("discriminator", [True, False])
@pytest.mark.parametrize("extension", ["safetensors", "pth"])
def test_retention_keeps_three_regular_sets_and_preserves_manual_and_backup_files(
    tmp_path: Path, ema: bool, discriminator: bool, extension: str
) -> None:
    layout = _layout(
        tmp_path, ema=ema, discriminator=discriminator, extension=extension
    )
    old = _checkpoint(layout, 183_000)
    first = _checkpoint(layout, 184_000)
    second = _checkpoint(layout, 185_000)
    manual = [_checkpoint(layout, i) for i in (183_543, 185_542, 185_543, 186_543)]
    backup = layout.states / "183000.state.bak"
    backup.write_text("manual backup")
    pretrained = layout.models / "pretrained.safetensors"
    _safetensors(pretrained)
    before = layout.before_save(186_000)
    latest = _checkpoint(layout, 186_000)
    old_bytes = sum(path.stat().st_size for path in layout.files(old))
    result = CheckpointRetention(layout, keep=3).prune_after_save(186_000, before)
    assert result.error is None
    assert result.removed_iterations == (183_000,)
    assert result.removed_files == tuple(layout.files(old))
    assert result.removed_bytes == old_bytes
    assert not any(path.exists() for path in layout.files(old))
    assert all(
        layout.complete(checkpoint) for checkpoint in [first, second, latest, *manual]
    )
    assert backup.read_text() == "manual backup"
    assert pretrained.exists()


def test_retention_does_not_prune_during_startup_or_count_future_saves(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    old = [_checkpoint(layout, i) for i in (181_000, 182_000, 183_000)]
    kept = [_checkpoint(layout, i) for i in (184_000, 185_000, 187_000, 188_000)]
    retention = CheckpointRetention(layout, keep=3)
    assert all(layout.complete(checkpoint) for checkpoint in old)
    before = layout.before_save(186_000)
    kept.append(_checkpoint(layout, 186_000))
    result = retention.prune_after_save(186_000, before)
    assert result.error is None
    assert result.removed_iterations == (181_000, 182_000, 183_000)
    assert all(layout.complete(checkpoint) for checkpoint in kept)


@pytest.mark.parametrize("iteration", [186_543, 0, -1])
def test_manual_and_final_saves_never_trigger_retention(
    tmp_path: Path, iteration: int
) -> None:
    layout = _layout(tmp_path)
    old = [_checkpoint(layout, i) for i in (182_000, 183_000, 184_000, 185_000)]
    before = layout.before_save(iteration)
    _checkpoint(layout, iteration)
    result = CheckpointRetention(layout, keep=3).prune_after_save(iteration, before)
    assert not result.removed_files
    assert all(layout.complete(checkpoint) for checkpoint in old)


@pytest.mark.parametrize("missing_file", [0, 1, 2])
def test_incomplete_new_checkpoint_never_retires_an_old_one(
    tmp_path: Path, missing_file: int
) -> None:
    layout = _layout(tmp_path)
    old = [_checkpoint(layout, i) for i in (183_000, 184_000, 185_000)]
    before = layout.before_save(186_000)
    latest = _checkpoint(layout, 186_000)
    layout.files(latest)[missing_file].unlink()
    result = CheckpointRetention(layout, keep=3).prune_after_save(186_000, before)
    assert result.error and "incomplete" in result.error
    assert not result.removed_files
    assert all(layout.complete(checkpoint) for checkpoint in old)


def test_stale_existing_save_never_triggers_retention(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    old = [_checkpoint(layout, i) for i in (183_000, 184_000, 185_000, 186_000)]
    before = layout.before_save(186_000)
    result = CheckpointRetention(layout, keep=3).prune_after_save(186_000, before)
    assert result.error and "not fully refreshed" in result.error
    assert not result.removed_files
    assert all(layout.complete(checkpoint) for checkpoint in old)


def test_partial_older_sets_do_not_count_as_recovery_buffers(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    kept = [_checkpoint(layout, i) for i in (183_000, 185_000)]
    partial = _checkpoint(layout, 184_000)
    layout.files(partial)[1].write_bytes(b"incomplete")
    before = layout.before_save(186_000)
    kept.append(_checkpoint(layout, 186_000))
    result = CheckpointRetention(layout, keep=3).prune_after_save(186_000, before)
    assert not result.removed_files
    assert all(layout.complete(checkpoint) for checkpoint in kept)
    assert partial.state_path.exists()


def _ready_retention(
    tmp_path: Path,
) -> tuple[CheckpointRetention, list[RecoveryCheckpoint], dict]:
    layout = _layout(tmp_path)
    checkpoints = [_checkpoint(layout, i) for i in (183_000, 184_000, 185_000)]
    before = layout.before_save(186_000)
    checkpoints.append(_checkpoint(layout, 186_000))
    return CheckpointRetention(layout, keep=3), checkpoints, before


def test_disk_flush_failure_preserves_every_checkpoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    retention, checkpoints, before = _ready_retention(tmp_path)
    monkeypatch.setattr(
        recovery_runtime.os, "fsync", Mock(side_effect=OSError("flush failed"))
    )
    result = retention.prune_after_save(186_000, before)
    assert result.error and "flush failed" in result.error
    assert not result.removed_files
    assert all(retention.layout.complete(checkpoint) for checkpoint in checkpoints)


def test_retained_sets_are_flushed_before_the_first_deletion(
    tmp_path: Path, monkeypatch: Any
) -> None:
    retention, checkpoints, before = _ready_retention(tmp_path)
    flushed: list[Path] = []
    original_unlink = Path.unlink

    def flush(files: list[Path]) -> None:
        assert all(retention.layout.complete(checkpoint) for checkpoint in checkpoints)
        flushed.extend(files)

    def unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        assert set(flushed) == {
            file
            for checkpoint in checkpoints[1:]
            for file in retention.layout.files(checkpoint)
        }
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(retention, "_flush_retained", flush)
    monkeypatch.setattr(Path, "unlink", unlink)
    result = retention.prune_after_save(186_000, before)
    assert result.removed_iterations == (183_000,)


def test_files_changed_during_flush_cancel_pruning(
    tmp_path: Path, monkeypatch: Any
) -> None:
    retention, checkpoints, before = _ready_retention(tmp_path)

    def flush(files: list[Path]) -> None:
        files[0].write_bytes(files[0].read_bytes() + b"changed concurrently")

    monkeypatch.setattr(retention, "_flush_retained", flush)
    result = retention.prune_after_save(186_000, before)
    assert result.error and "changed during verification" in result.error
    assert not result.removed_files
    assert all(checkpoint.state_path.exists() for checkpoint in checkpoints)


@pytest.mark.parametrize("redirect", ["state", "weights", "directory"])
def test_redirected_checkpoint_paths_are_never_pruned(
    tmp_path: Path, monkeypatch: Any, redirect: str
) -> None:
    retention, checkpoints, before = _ready_retention(tmp_path)
    if redirect == "directory":
        directory = tmp_path / "external-models"
        retention.layout.models.rename(directory)
        retention.layout.models.symlink_to(directory, target_is_directory=True)
    else:
        old_path = retention.layout.files(checkpoints[0])[
            0 if redirect == "state" else 1
        ]
        external = tmp_path / f"external{old_path.suffix}"
        old_path.rename(external)
        old_path.symlink_to(external)
    flush = Mock()
    monkeypatch.setattr(retention, "_flush_retained", flush)
    result = retention.prune_after_save(186_000, before)
    assert result.error and "Refusing retention" in result.error
    assert not result.removed_files
    flush.assert_not_called()
    assert all(retention.layout.complete(checkpoint) for checkpoint in checkpoints)


def test_pretrained_source_inside_numbered_set_is_protected(tmp_path: Path) -> None:
    retention, checkpoints, before = _ready_retention(tmp_path)
    old = checkpoints[0]
    protected = CheckpointRetention(
        retention.layout, 3, (retention.layout.files(old)[1],)
    )
    result = protected.prune_after_save(186_000, before)
    assert not result.removed_files
    assert all(retention.layout.complete(checkpoint) for checkpoint in checkpoints)


def test_unlink_failure_never_harms_the_three_retained_sets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    retention, checkpoints, before = _ready_retention(tmp_path)
    original_unlink = Path.unlink
    failed_path = retention.layout.files(checkpoints[0])[1]

    def unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == failed_path:
            raise PermissionError("simulated deletion failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    result = retention.prune_after_save(186_000, before)
    assert result.error and "deletion failure" in result.error
    assert result.removed_files == (checkpoints[0].state_path,)
    assert not checkpoints[0].state_path.exists()
    assert failed_path.exists()
    assert all(retention.layout.complete(checkpoint) for checkpoint in checkpoints[1:])


def test_pruning_preserves_the_supervisors_rollback_anchor(
    tmp_path: Path, monkeypatch: Any
) -> None:
    retention, checkpoints, before = _ready_retention(tmp_path)
    recovery = TrainingRecovery(retention.layout, tmp_path / "failure.json")
    recovery.note_resume(str(checkpoints[2].state_path), 185_000)
    assert recovery.note_periodic_save(186_000, before)
    result = retention.prune_after_save(186_000, before)
    assert result.removed_iterations == (183_000,)
    _checkpoint(retention.layout, 186_543)
    recovery.report_failure(
        PersistentNonfiniteError("nonfinite_generator_loss", 186_600, 20), 7970
    )
    report = json.loads(recovery.report_path.read_text())
    assert report["checkpoint"]["iteration"] == 186_000
    monkeypatch.setenv(RESUME_ENV, report["checkpoint"]["state_path"])
    options = SimpleNamespace(auto_resume=True, path=SimpleNamespace(resume_state=None))
    assert recovery.choose_resume(options) == str(checkpoints[-1].state_path)


def test_retention_is_opt_in_and_rank_zero_only(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    options = SimpleNamespace(
        rank=0,
        network_g={},
        network_d=None,
        train=SimpleNamespace(ema_decay=0.999),
        logger=SimpleNamespace(
            save_checkpoint_keep=0,
            save_checkpoint_freq=1000,
            save_checkpoint_format="safetensors",
        ),
        path=SimpleNamespace(
            training_states=str(layout.states),
            models=str(layout.models),
            resume_models=str(layout.resume_models),
        ),
    )
    assert TrainingRecovery.from_environment(options) is None
    assert CheckpointRetention.from_options(options) is None
    options.logger.save_checkpoint_keep = 3
    retention = CheckpointRetention.from_options(options)
    assert retention is not None and retention.keep == 3 and retention.layout == layout
    options.rank = 1
    assert CheckpointRetention.from_options(options) is None


def test_retention_option_defaults_off_and_rejects_negative_values() -> None:
    import msgspec
    from traiNNer.utils.redux_options import LogOptions

    config = {"print_freq": 100, "save_checkpoint_freq": 1000, "use_tb_logger": True}
    assert msgspec.convert(config, type=LogOptions).save_checkpoint_keep == 0
    assert (
        msgspec.convert(
            {**config, "save_checkpoint_keep": 3}, type=LogOptions
        ).save_checkpoint_keep
        == 3
    )
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({**config, "save_checkpoint_keep": -1}, type=LogOptions)


@pytest.mark.parametrize("supervised", [True, False])
@pytest.mark.parametrize("complete_save", [True, False])
def test_real_training_save_hook_prunes_only_after_a_complete_save(
    tmp_path: Path, monkeypatch: Any, supervised: bool, complete_save: bool
) -> None:
    # Exercise the real orchestration with one accepted fake iteration. Every
    # model, data-loader and CUDA operation is replaced by CPU-only fixtures.
    import train as entry

    layout = _layout(tmp_path)
    old = _checkpoint(layout, 183_000)
    kept = [_checkpoint(layout, i) for i in (184_000, 185_000, 185_543)]
    options = SimpleNamespace(
        name="retention_integration",
        root_path=str(tmp_path),
        manual_seed=7970,
        rank=0,
        network_g={},
        network_d=None,
        train=SimpleNamespace(gpu_lease=None, ema_decay=0.999, warmup_iter=-1),
        logger=SimpleNamespace(
            use_tb_logger=False,
            save_checkpoint_keep=3,
            save_checkpoint_freq=1000,
            save_checkpoint_format="safetensors",
            print_freq=100,
        ),
        path=SimpleNamespace(
            experiments_root=str(tmp_path),
            log=str(tmp_path),
            models=str(layout.models),
            resume_models=str(layout.resume_models),
            training_states=str(layout.states),
        ),
        detect_anomaly=False,
        deterministic=False,
        contents="",
        val=None,
        fast_matmul=False,
        dist=False,
        datasets={"train": SimpleNamespace(prefetch_mode="cpu")},
    )
    recovery = (
        TrainingRecovery(layout, tmp_path / "failure.json") if supervised else None
    )
    model = Mock(
        with_metrics=False,
        accum_iters=1,
        opt=options,
        training_step_accepted=True,
        reset_accumulation_requested=False,
    )
    model.data_feed_context.return_value = nullcontext()
    model.get_current_log.return_value = {}

    def save(epoch: int, current_iter: int) -> None:
        if current_iter == -1:
            return  # Final latest save is outside the retention path.
        assert current_iter == 186_000
        assert layout.complete(old), "Pruning happened before the save"
        latest = _checkpoint(layout, current_iter)
        if not complete_save:
            layout.files(latest)[-1].unlink()

    model.save.side_effect = save
    prefetcher = Mock()
    prefetcher.next.side_effect = [{"fake": "CPU fixture"}, None]
    monkeypatch.setattr(
        entry.Config,
        "load_config_from_file",
        Mock(return_value=(options, SimpleNamespace(opt="unused.yml"))),
    )
    monkeypatch.setattr(
        entry.TrainingRecovery, "from_environment", Mock(return_value=recovery)
    )
    monkeypatch.setattr(
        entry, "load_resume_state", Mock(return_value={"epoch": 8, "iter": 185_999})
    )
    monkeypatch.setattr(entry, "build_model", Mock(return_value=model))
    monkeypatch.setattr(
        entry,
        "create_train_val_dataloader",
        Mock(return_value=(Mock(), Mock(), [], 8, 186_000)),
    )
    monkeypatch.setattr(entry, "init_tb_loggers", Mock(return_value=Mock()))
    monkeypatch.setattr(entry, "CPUPrefetcher", Mock(return_value=prefetcher))
    monkeypatch.setattr(
        entry, "MessageLogger", Mock(return_value=Mock(start_iter=185_999))
    )
    monkeypatch.setattr(entry, "get_root_logger", Mock(return_value=Mock()))
    monkeypatch.setattr(
        entry, "get_env_info", Mock(return_value="CPU checkpoint fixture")
    )
    monkeypatch.setattr(entry, "diff_user_vs_template", Mock(return_value=(None, None)))
    for name in ("install", "set_random_seed", "make_exp_dirs", "copy_opt_file"):
        monkeypatch.setattr(entry, name, Mock())
    monkeypatch.setattr(entry.torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(entry.torch.cuda, "set_per_process_memory_fraction", Mock())
    monkeypatch.setattr(entry.torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(entry.signal, "signal", Mock())
    entry.train_pipeline(str(tmp_path))
    model.optimize_parameters.assert_called_once_with(186_000, 0, True)
    assert old.state_path.exists() is not complete_save
    assert all(layout.complete(checkpoint) for checkpoint in kept)
    if recovery is not None and complete_save:
        assert recovery.checkpoint and recovery.checkpoint.iteration == 186_000


_WORKER = """
import json, os, signal, sys, time
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
calls = root / "calls.jsonl"
previous = calls.read_text().splitlines() if calls.exists() else []
with calls.open("a") as stream:
    stream.write(json.dumps({"resume": os.environ.get("TRAINNER_RECOVERY_RESUME_STATE"),
                             "seed": os.environ.get("TRAINNER_RECOVERY_MANUAL_SEED")}) + "\\n")
if mode == "recover" and previous:
    sys.exit(0)
if mode == "generic":
    sys.exit(1)
if mode == "normal":
    sys.exit(0)
if mode == "no_report":
    sys.exit(75)

if mode == "manual":
    def interrupt(signum, frame):
        (root / "manual-stop").write_text("stopped")
        sys.exit(0)
    signal.signal(signal.SIGINT, interrupt)
    (root / "ready").write_text("ready")
    while True:
        time.sleep(0.05)

report = {"version": 1, "kind": "persistent_nonfinite", "pid": os.getpid(),
          "reason": "nonfinite_generator_loss", "candidate_iter": 176143,
          "consecutive_rejections": 20, "manual_seed": 7970,
          "experiment_dir": str(root),
          "checkpoint": {"iteration": 176000,
                         "state_path": str(root / "training_states" / "176000.state")}}
if mode == "stale":
    report["pid"] += 1
if mode == "future":
    report["checkpoint"]["iteration"] = 177000
if mode == "no_checkpoint":
    report["checkpoint"] = None
Path(os.environ["TRAINNER_NONFINITE_REPORT"]).write_text(json.dumps(report))
sys.exit(75)
"""


def _worker(tmp_path: Path, mode: str) -> list[str]:
    _checkpoint(_layout(tmp_path), 176_000)
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    return [sys.executable, str(script), str(tmp_path), mode]


def _normal_command_worker(tmp_path: Path, mode: str) -> list[str]:
    # Run the production entrypoint bootstrap with a tiny CPU fault fixture in
    # place of training. The parent re-executes this exact same entrypoint.
    _checkpoint(_layout(tmp_path), 176_000)
    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "train.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        + _entrypoint_bootstrap()
        + "\nassert 'torch' not in sys.modules\n"
        + _WORKER
    )
    return [
        sys.executable,
        str(script),
        str(tmp_path),
        mode,
        "-opt",
        "unused.yml",
        "--recovery-delay",
        "0",
    ]


@pytest.mark.parametrize(
    ("mode", "exit_code", "launches"),
    [("recover", 0, 2), ("always_fail", 75, 3), ("generic", 1, 1), ("normal", 0, 1)],
)
def test_normal_command_end_to_end_recovery_without_a_separate_launcher(
    tmp_path: Path, mode: str, exit_code: int, launches: int
) -> None:
    result = subprocess.run(
        _normal_command_worker(tmp_path, mode),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == exit_code, (result.stdout, result.stderr)
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert len(calls) == launches
    if mode == "recover":
        assert calls[-1] == {
            "resume": str(tmp_path / "training_states" / "176000.state"),
            "seed": "7970",
        }
        assert "176,143 -> 176,000" in result.stdout


def test_normal_command_opt_out_does_not_restart_a_nonfinite_exit(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [*_normal_command_worker(tmp_path, "no_report"), "--no-auto-recover"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == NONFINITE_EXIT_CODE, (result.stdout, result.stderr)
    assert "[recovery]" not in result.stdout
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1


def test_real_subprocess_rolls_back_176143_to_176000(tmp_path: Path) -> None:
    command = _worker(tmp_path, "recover")
    assert supervise(command, restart_delay=0) == 0
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert calls == [
        {"resume": None, "seed": None},
        {"resume": str(tmp_path / "training_states" / "176000.state"), "seed": "7970"},
    ]
    history = json.loads((tmp_path / "training_recovery.jsonl").read_text())
    assert history["recovery_action"] == "restart"


def test_real_subprocess_stops_after_two_no_progress_restarts(tmp_path: Path) -> None:
    command = _worker(tmp_path, "always_fail")
    assert supervise(command, restart_delay=0) == NONFINITE_EXIT_CODE
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 3
    history = [
        json.loads(line)
        for line in (tmp_path / "training_recovery.jsonl").read_text().splitlines()
    ]
    assert history[-1]["recovery_action"] == "stop_no_progress"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("normal", 0),
        ("generic", 1),
        ("no_report", 75),
        ("stale", 75),
        ("future", 75),
        ("no_checkpoint", 75),
    ],
)
def test_real_subprocess_does_not_restart_other_exits_or_invalid_reports(
    tmp_path: Path, mode: str, expected: int
) -> None:
    assert supervise(_worker(tmp_path, mode), restart_delay=0) == expected
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1


def test_normal_command_manual_ctrl_c_is_forwarded_and_never_restarted(
    tmp_path: Path,
) -> None:
    process = subprocess.Popen(
        _normal_command_worker(tmp_path, "manual"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready").exists() and process.poll() is None:
            if time.monotonic() > deadline:
                pytest.fail("Fake trainer did not become ready")
            time.sleep(0.05)
        assert process.poll() is None
        process.send_signal(signal.SIGINT)
        output, error = process.communicate(timeout=10)
        assert process.returncode == 130, (output, error)
        assert (tmp_path / "manual-stop").read_text() == "stopped"
        assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


def test_training_without_a_supervisor_does_not_modify_default_seed(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv(REPORT_ENV, raising=False)
    monkeypatch.setenv(SEED_ENV, "42")
    options = SimpleNamespace(manual_seed=7970)
    assert TrainingRecovery.from_environment(options) is None
    assert options.manual_seed == 7970


@pytest.mark.parametrize(
    "reason",
    [
        "nonfinite_generator_loss",
        "nonfinite_generator_gradient",
        "nonfinite_guard_sample",
    ],
)
def test_specific_exception_preserves_failure_kind(reason: str) -> None:
    error = PersistentNonfiniteError(reason, 176_143, 20)
    assert isinstance(error, RuntimeError)
    assert error.reason == reason
    assert "20 consecutive" in str(error)
    assert "176,143" in str(error)


@pytest.mark.parametrize(
    "reason", ["nonfinite_generator_loss", "nonfinite_generator_gradient"]
)
def test_real_training_loop_does_not_save_after_persistent_nonfinite(
    tmp_path: Path, monkeypatch: Any, reason: str
) -> None:
    # Execute the real exception/cleanup path with every CUDA operation and
    # data/model construction replaced. No training tensors or GPU calls.
    import train as entry

    options = SimpleNamespace(
        name="fault_injection",
        root_path=str(tmp_path),
        manual_seed=7970,
        rank=0,
        train=SimpleNamespace(gpu_lease=None),
        logger=SimpleNamespace(use_tb_logger=False, save_checkpoint_keep=0),
        path=SimpleNamespace(
            experiments_root=str(tmp_path), log=str(tmp_path), models=str(tmp_path)
        ),
        detect_anomaly=False,
        deterministic=False,
        contents="",
        val=None,
        fast_matmul=False,
        dist=False,
        datasets={"train": SimpleNamespace(prefetch_mode="cpu")},
    )
    failure = PersistentNonfiniteError(reason, 176_143, 20)
    recovery = Mock()
    tensorboard = Mock()
    model = Mock(with_metrics=False, accum_iters=1, opt=options)
    model.data_feed_context.return_value = nullcontext()
    model.optimize_parameters.side_effect = failure
    prefetcher = Mock()
    prefetcher.next.return_value = {"fake": "CPU fixture"}
    monkeypatch.setattr(
        entry.Config,
        "load_config_from_file",
        Mock(return_value=(options, SimpleNamespace(opt="unused.yml"))),
    )
    monkeypatch.setattr(
        entry.TrainingRecovery, "from_environment", Mock(return_value=recovery)
    )
    monkeypatch.setattr(
        entry, "load_resume_state", Mock(return_value={"epoch": 8, "iter": 176_000})
    )
    monkeypatch.setattr(entry, "build_model", Mock(return_value=model))
    monkeypatch.setattr(
        entry,
        "create_train_val_dataloader",
        Mock(return_value=(Mock(), Mock(), [], 8, 800_000)),
    )
    monkeypatch.setattr(entry, "init_tb_loggers", Mock(return_value=tensorboard))
    monkeypatch.setattr(entry, "CPUPrefetcher", Mock(return_value=prefetcher))
    monkeypatch.setattr(entry, "MessageLogger", Mock())
    monkeypatch.setattr(entry, "get_root_logger", Mock(return_value=Mock()))
    monkeypatch.setattr(entry, "get_env_info", Mock(return_value="CPU fault injection"))
    monkeypatch.setattr(entry, "diff_user_vs_template", Mock(return_value=(None, None)))
    for name in ("install", "set_random_seed", "make_exp_dirs", "copy_opt_file"):
        monkeypatch.setattr(entry, name, Mock())
    monkeypatch.setattr(entry.torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(entry.torch.cuda, "set_per_process_memory_fraction", Mock())
    monkeypatch.setattr(entry.torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(entry.signal, "signal", Mock())
    with pytest.raises(SystemExit) as stopped:
        entry.train_pipeline(str(tmp_path))
    assert stopped.value.code == NONFINITE_EXIT_CODE
    model.save.assert_not_called()
    recovery.report_failure.assert_called_once_with(failure, 7970)
    tensorboard.close.assert_called_once()
