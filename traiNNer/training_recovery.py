"""Failure-only checkpoint recovery; no tensor work or per-step monitoring.

The trainer publishes a small report only on a persistent nonfinite failure.
The train.py parent waits for that process to exit before creating its replacement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import FrameType

    from traiNNer.utils.redux_options import ReduxOptions


MAX_CONSECUTIVE_NONFINITE_ATTEMPTS = 20
NONFINITE_EXIT_CODE = 75
REPORT_ENV = "TRAINNER_NONFINITE_REPORT"
RESUME_ENV = "TRAINNER_RECOVERY_RESUME_STATE"
SEED_ENV = "TRAINNER_RECOVERY_MANUAL_SEED"
_REASONS = {
    "nonfinite_generator_loss": "generator losses",
    "nonfinite_generator_gradient": "generator gradients",
    "nonfinite_guard_sample": "attempts",
}


def add_recovery_arguments(parser: argparse.ArgumentParser) -> None:
    """Share options between the lightweight bootstrap and normal training help."""
    group = parser.add_argument_group("automatic NaN/Inf recovery")
    group.add_argument(
        "--no-auto-recover",
        action="store_true",
        help="Disable the default single-rank WSL/Linux recovery supervisor.",
    )
    group.add_argument(
        "--recovery-max-restarts",
        type=int,
        default=2,
        help="Maximum restarts without a newer complete checkpoint (default: 2).",
    )
    group.add_argument(
        "--recovery-delay",
        type=float,
        default=5.0,
        help="Seconds to wait before restarting after a nonfinite failure (default: 5).",
    )


def maybe_supervise_training(script_path: str, argv: list[str] | None = None) -> None:
    """Supervise the normal CLI before Torch imports; workers run it directly."""
    if os.environ.get(REPORT_ENV):
        # The supervisor already owns this worker. Never recursively launch.
        return
    training_args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in training_args) or not any(
        arg == "-opt" or arg.startswith("-opt=") for arg in training_args
    ):
        # Keep help and missing-option errors on the existing CLI path.
        return
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_recovery_arguments(parser)
    parser.add_argument("--launcher", default="none")
    args, _ = parser.parse_known_args(training_args)
    if args.no_auto_recover:
        return
    if (
        args.recovery_max_restarts < 1
        or not math.isfinite(args.recovery_delay)
        or args.recovery_delay < 0
    ):
        parser.error("Restart count must be positive and delay finite/non-negative.")
    if os.name != "posix":
        print(
            "[recovery] Automatic restart requires native WSL/Linux; "
            "continuing with ordinary training.",
            flush=True,
        )
        return
    if (
        args.launcher != "none"
        or os.environ.get("WORLD_SIZE", "1") != "1"
        or any(name in os.environ for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"))
    ):
        # Restarting an individual rank can hang the other ranks. Distributed
        # launchers retain ownership until coordinated recovery is implemented.
        print(
            "[recovery] Automatic restart is disabled for distributed launchers; "
            "continuing with ordinary training.",
            flush=True,
        )
        return
    raise SystemExit(
        supervise(
            [sys.executable, "-u", str(Path(script_path).resolve()), *training_args],
            max_restarts=args.recovery_max_restarts,
            restart_delay=args.recovery_delay,
        )
    )


class PersistentNonfiniteError(RuntimeError):
    """A typed, restartable failure, distinct from CUDA/OOM/other exceptions."""

    def __init__(self, reason: str, candidate_iter: int, count: int) -> None:
        if reason not in _REASONS:
            raise ValueError(f"Unknown nonfinite rejection reason: {reason}")
        self.reason = reason
        self.candidate_iter = candidate_iter
        self.count = count
        super().__init__(
            f"Training failed: reached the limit of {count} consecutive NaN/Inf "
            f"rejected {_REASONS[reason]} at attempted iteration {candidate_iter:,}."
        )


def _complete_tensor_file(path: Path) -> bool:
    """Check container completeness without loading tensors or scanning payloads."""
    try:
        stat = path.stat()
        if stat.st_size <= 8:
            return False
        if path.suffix == ".safetensors":
            with path.open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                if not 0 < header_size <= min(stat.st_size - 8, 100_000_000):
                    return False
                header = json.loads(stream.read(header_size))
            if not isinstance(header, dict):
                return False
            offsets = [
                value["data_offsets"]
                for key, value in header.items()
                if key != "__metadata__"
            ]
            return bool(offsets) and max(end for _, end in offsets) == (
                stat.st_size - 8 - header_size
            )
        # Current torch.save writes ZIP archives; the central directory is only
        # finalized at the end of a successful write. Do not read large storages.
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return any(name.endswith("/data.pkl") for name in names)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return False


def _file_version(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class RecoveryCheckpoint:
    iteration: int
    state_path: Path


@dataclass(frozen=True)
class CheckpointLayout:
    states: Path
    models: Path
    resume_models: Path
    interval: int
    extension: str
    ema: bool
    discriminator: bool

    @classmethod
    def from_options(cls, opt: ReduxOptions) -> CheckpointLayout:
        assert opt.train is not None
        assert opt.logger is not None
        assert opt.path.training_states is not None
        assert opt.path.models is not None
        assert opt.path.resume_models is not None
        return cls(
            states=Path(opt.path.training_states).resolve(),
            models=Path(opt.path.models).resolve(),
            resume_models=Path(opt.path.resume_models).resolve(),
            interval=opt.logger.save_checkpoint_freq,
            extension=opt.logger.save_checkpoint_format,
            ema=opt.train.ema_decay > 0,
            discriminator=opt.network_d is not None,
        )

    def files(self, checkpoint: RecoveryCheckpoint) -> list[Path]:
        suffix = f"{checkpoint.iteration}.{self.extension}"
        raw_directory = self.resume_models if self.ema else self.models
        paths = [checkpoint.state_path, raw_directory / f"net_g_{suffix}"]
        if self.ema:
            paths.append(self.models / f"net_g_ema_{suffix}")
        if self.discriminator:
            paths.append(self.resume_models / f"net_d_{suffix}")
        return paths

    def complete(self, checkpoint: RecoveryCheckpoint) -> bool:
        return all(_complete_tensor_file(path) for path in self.files(checkpoint))

    def completed(
        self, *, regular_only: bool = False, through: int | None = None
    ) -> Iterator[RecoveryCheckpoint]:
        """Newest-first complete sets; never treat backups as numbered saves."""
        if regular_only and self.interval <= 0:
            return
        iterations = sorted(
            (
                int(path.stem)
                for path in self.states.glob("*.state")
                if path.stem.isascii()
                and path.stem.isdecimal()
                and int(path.stem) > 0
                and path.stem == str(int(path.stem))
                and (not regular_only or int(path.stem) % self.interval == 0)
                and (through is None or int(path.stem) <= through)
            ),
            reverse=True,
        )
        for iteration in iterations:
            candidate = RecoveryCheckpoint(
                iteration, self.states / f"{iteration}.state"
            )
            if self.complete(candidate):
                yield candidate

    def latest_regular(self) -> RecoveryCheckpoint | None:
        return next(self.completed(regular_only=True), None)

    def latest_complete(self) -> RecoveryCheckpoint | None:
        return next(self.completed(), None)

    def before_save(
        self, iteration: int
    ) -> dict[Path, tuple[int, int, int, int] | None]:
        checkpoint = RecoveryCheckpoint(iteration, self.states / f"{iteration}.state")
        return {path: _file_version(path) for path in self.files(checkpoint)}

    def fresh_save_complete(
        self, iteration: int, previous: dict[Path, tuple[int, int, int, int] | None]
    ) -> bool:
        checkpoint = RecoveryCheckpoint(iteration, self.states / f"{iteration}.state")
        files = self.files(checkpoint)
        return (
            set(previous) == set(files)
            and all(_file_version(path) != previous[path] for path in files)
            and self.complete(checkpoint)
        )


@dataclass(frozen=True)
class CheckpointPruneResult:
    removed_iterations: tuple[int, ...] = ()
    removed_files: tuple[Path, ...] = ()
    removed_bytes: int = 0
    error: str | None = None


class CheckpointRetention:
    """Opt-in retention of complete regular saves, independent of supervision."""

    def __init__(
        self,
        layout: CheckpointLayout,
        keep: int,
        protected_files: tuple[Path, ...] = (),
    ) -> None:
        if keep < 1 or layout.interval < 1:
            raise ValueError(
                "Checkpoint retention requires positive keep and interval."
            )
        self.layout = layout
        self.keep = keep
        self.protected_files = frozenset(path.resolve() for path in protected_files)

    @classmethod
    def from_options(cls, opt: ReduxOptions) -> CheckpointRetention | None:
        assert opt.logger is not None
        keep = opt.logger.save_checkpoint_keep
        if keep == 0 or opt.rank != 0:
            return None
        if opt.network_g is None:
            raise ValueError("Checkpoint retention supports SR/GAN checkpoint sets.")
        protected = tuple(
            Path(value).expanduser().resolve()
            for key in (
                "pretrain_network_g",
                "pretrain_network_g_ema",
                "pretrain_network_g_teacher",
                "pretrain_network_d",
            )
            if (value := getattr(opt.path, key, None))
        )
        return cls(CheckpointLayout.from_options(opt), keep, protected)

    def _validate_layout(self) -> None:
        root = self.layout.states.parent
        expected = (
            root / "training_states",
            root / "models",
            root / "models/resume_models",
        )
        actual = (self.layout.states, self.layout.models, self.layout.resume_models)
        if actual != expected or any(path.resolve() != path for path in actual):
            raise ValueError(
                "Refusing retention outside the experiment checkpoint folders."
            )

    def _owned_files(self, checkpoint: RecoveryCheckpoint) -> list[Path]:
        files = self.layout.files(checkpoint)
        for path in files:
            if path.is_symlink() or not path.is_file() or path.resolve() != path:
                raise ValueError(
                    f"Refusing retention of a missing or redirected file: {path}"
                )
        return files

    @staticmethod
    def _flush_retained(files: list[Path]) -> None:
        # Container validity alone does not make a save durable after a power
        # cut. Flush the retained sets before any old recovery point is removed.
        for path in files:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        if os.name == "posix":
            for directory in sorted({path.parent for path in files}):
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)

    def prune_after_save(
        self, iteration: int, previous: dict[Path, tuple[int, int, int, int] | None]
    ) -> CheckpointPruneResult:
        """Only a freshly completed periodic save may trigger deletion.

        Manual/non-periodic saves never count toward the limit and are never
        candidates. A deliberate rollback also leaves all future saves alone.
        """
        removed_iterations: list[int] = []
        removed_files: list[Path] = []
        removed_bytes = 0
        try:
            if iteration <= 0 or iteration % self.layout.interval:
                return CheckpointPruneResult()
            self._validate_layout()
            if not self.layout.fresh_save_complete(iteration, previous):
                return CheckpointPruneResult(
                    error="New checkpoint is incomplete or not fully refreshed; nothing pruned."
                )
            checkpoints = list(
                self.layout.completed(regular_only=True, through=iteration)
            )
            if len(checkpoints) <= self.keep:
                return CheckpointPruneResult()
            retained = checkpoints[: self.keep]
            if retained[0].iteration != iteration:
                return CheckpointPruneResult(
                    error="New checkpoint disappeared; nothing pruned."
                )
            candidates = [
                checkpoint
                for checkpoint in reversed(checkpoints[self.keep :])
                if not any(
                    path.resolve() in self.protected_files
                    for path in self.layout.files(checkpoint)
                )
            ]
            if not candidates:
                return CheckpointPruneResult()
            retained_files = [
                path
                for checkpoint in retained
                for path in self._owned_files(checkpoint)
            ]
            retired_files = [
                path
                for checkpoint in candidates
                for path in self._owned_files(checkpoint)
            ]
            versions = {
                path: _file_version(path) for path in retained_files + retired_files
            }
            self._flush_retained(retained_files)
            if any(
                _file_version(path) != version for path, version in versions.items()
            ):
                return CheckpointPruneResult(
                    error="Checkpoint files changed during verification; nothing pruned."
                )
            self._validate_layout()
            for checkpoint in candidates:
                # State first: if deletion is interrupted, the leftover weights
                # cannot be mistaken for another complete resumable checkpoint.
                for path in self._owned_files(checkpoint):
                    version = versions[path]
                    assert version is not None
                    path.unlink()
                    removed_files.append(path)
                    removed_bytes += version[1]
                removed_iterations.append(checkpoint.iteration)
        except (OSError, ValueError) as error:
            return CheckpointPruneResult(
                tuple(removed_iterations),
                tuple(removed_files),
                removed_bytes,
                str(error),
            )
        return CheckpointPruneResult(
            tuple(removed_iterations), tuple(removed_files), removed_bytes
        )


class TrainingRecovery:
    """Bookkeeping at startup/checkpoint/failure boundaries, never per iteration."""

    def __init__(self, layout: CheckpointLayout, report_path: Path) -> None:
        self.layout = layout
        self.report_path = report_path
        self.checkpoint: RecoveryCheckpoint | None = None

    @classmethod
    def from_environment(cls, opt: ReduxOptions) -> TrainingRecovery | None:
        report_path = os.environ.get(REPORT_ENV)
        if not report_path:
            return None
        if opt.dist or opt.network_g is None:
            raise ValueError(
                "Automatic nonfinite recovery supports single-rank SR training."
            )
        if opt.path.ignore_resume_networks:
            raise ValueError(
                "Recovery must restore every checkpoint network together. "
                "Use --no-auto-recover for a partial restore."
            )
        assert opt.train is not None
        assert opt.logger is not None
        if opt.train.restart_scheduler:
            raise ValueError(
                "Automatic recovery requires restart_scheduler: false. "
                "Use --no-auto-recover to deliberately restart the scheduler."
            )
        if opt.logger.save_checkpoint_freq <= 0:
            raise ValueError(
                "Automatic recovery requires regular checkpoint saving. "
                "Use --no-auto-recover to train without regular checkpoints."
            )
        layout = CheckpointLayout.from_options(opt)
        seed = os.environ.get(SEED_ENV)
        if seed is not None:
            opt.manual_seed = int(seed)
        return cls(layout, Path(report_path))

    def choose_resume(self, opt: ReduxOptions) -> str | None:
        forced = os.environ.get(RESUME_ENV)
        if forced:
            # This wins over --auto_resume: never accidentally select a newer
            # interrupt/crash save instead of the checkpoint chosen by recovery.
            return str(Path(forced).resolve())
        if not opt.auto_resume:
            return opt.path.resume_state
        checkpoint = self.layout.latest_complete()
        if checkpoint is None:
            if any(self.layout.states.glob("*.state")):
                raise RuntimeError(
                    "No complete checkpoint was found. Refusing to use an "
                    "incomplete save or start over; select a known-good path.resume_state."
                )
            return None
        return str(checkpoint.state_path)

    def note_resume(self, state_path: str, iteration: int) -> None:
        checkpoint = RecoveryCheckpoint(iteration, Path(state_path).resolve())
        if not self.layout.complete(checkpoint):
            raise RuntimeError(
                "Recovery requires a complete matching state/model/EMA checkpoint."
            )
        if (
            checkpoint.state_path.stem.isdecimal()
            and int(checkpoint.state_path.stem) != iteration
        ):
            raise RuntimeError("Checkpoint filename and saved iteration disagree.")
        # A complete manual-stop checkpoint is a valid startup anchor as well.
        # Failure retries use this anchor directly, never rescan for newer files.
        self.checkpoint = checkpoint

    def before_periodic_save(
        self, iteration: int
    ) -> dict[Path, tuple[int, int, int, int] | None]:
        return self.layout.before_save(iteration)

    def note_periodic_save(
        self, iteration: int, previous: dict[Path, tuple[int, int, int, int] | None]
    ) -> bool:
        checkpoint = RecoveryCheckpoint(
            iteration, self.layout.states / f"{iteration}.state"
        )
        if iteration % self.layout.interval or not self.layout.fresh_save_complete(
            iteration, previous
        ):
            return False
        self.checkpoint = checkpoint
        return True

    def report_failure(
        self, failure: PersistentNonfiniteError, manual_seed: int
    ) -> None:
        checkpoint = self.checkpoint
        if checkpoint is not None and not self.layout.complete(checkpoint):
            checkpoint = None
        report = {
            "version": 1,
            "kind": "persistent_nonfinite",
            "pid": os.getpid(),
            "reason": failure.reason,
            "candidate_iter": failure.candidate_iter,
            "consecutive_rejections": failure.count,
            "manual_seed": manual_seed,
            "experiment_dir": str(self.layout.states.parent),
            "checkpoint": None
            if checkpoint is None
            else {
                "iteration": checkpoint.iteration,
                "state_path": str(checkpoint.state_path),
            },
        }
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report), encoding="utf-8")
        temporary.replace(self.report_path)


@dataclass
class RecoveryBudget:
    max_restarts: int = 2
    checkpoint_iter: int = -1
    restarts: int = 0

    def allow(self, checkpoint_iter: int) -> bool:
        if checkpoint_iter > self.checkpoint_iter:
            self.checkpoint_iter = checkpoint_iter
            self.restarts = 0
        if self.restarts >= self.max_restarts:
            return False
        self.restarts += 1
        return True


def _read_report(path: Path, child_pid: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Invalid nonfinite failure report.")
    if (
        report.get("version") != 1
        or report.get("kind") != "persistent_nonfinite"
        or report.get("pid") != child_pid
        or report.get("reason") not in _REASONS
        or type(report.get("candidate_iter")) is not int
        or type(report.get("manual_seed")) is not int
        or report.get("consecutive_rejections") != MAX_CONSECUTIVE_NONFINITE_ATTEMPTS
        or not isinstance(report.get("experiment_dir"), str)
        or not Path(report["experiment_dir"]).is_absolute()
        or not Path(report["experiment_dir"]).is_dir()
    ):
        raise ValueError("Invalid or stale nonfinite failure report.")
    checkpoint = report.get("checkpoint")
    if checkpoint is None:
        raise ValueError("No complete checkpoint is available for recovery.")
    if (
        not isinstance(checkpoint, dict)
        or type(checkpoint.get("iteration")) is not int
        or not 0 < checkpoint["iteration"] < report["candidate_iter"]
        or not isinstance(checkpoint.get("state_path"), str)
        or not Path(checkpoint["state_path"]).is_absolute()
        or not _complete_tensor_file(Path(checkpoint["state_path"]))
    ):
        raise ValueError(
            "Recovery checkpoint is missing, incomplete, or not older than the failure."
        )
    return report


def supervise(
    command: list[str], *, max_restarts: int = 2, restart_delay: float = 5.0
) -> int:
    """Wait on a training process; restart only its typed nonfinite exit.

    Each child owns a process group so Ctrl+C reaches the trainer once, without
    interrupting its dataloader workers during a graceful checkpoint save.
    """
    if os.name != "posix":
        raise ValueError("Use this recovery launcher from native WSL/Linux.")
    if max_restarts < 1 or not math.isfinite(restart_delay) or restart_delay < 0:
        raise ValueError(
            "max_restarts must be positive and restart_delay finite/non-negative."
        )
    budget = RecoveryBudget(max_restarts)
    child: subprocess.Popen | None = None
    stop_requested = False
    next_state: str | None = None
    seed: int | None = None

    def stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stop_requested
        del signum, frame
        force = stop_requested
        stop_requested = True
        if child is not None and child.poll() is None:
            try:
                if force:
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with tempfile.TemporaryDirectory(prefix="trainner-recovery-") as temp:
            attempt = 0
            while not stop_requested:
                attempt += 1
                report_path = Path(temp) / f"failure-{attempt}.json"
                env = os.environ.copy()
                env[REPORT_ENV] = str(report_path)
                env.pop(RESUME_ENV, None)
                env.pop(SEED_ENV, None)
                if next_state is not None:
                    env[RESUME_ENV] = next_state
                    env[SEED_ENV] = str(seed)
                print(
                    f"[recovery] Starting training process (launch {attempt}).",
                    flush=True,
                )
                child = subprocess.Popen(command, env=env, start_new_session=True)
                if stop_requested:
                    child.send_signal(signal.SIGINT)
                child_pid = child.pid
                returncode = child.wait()
                # Clean up only this launched trainer's remaining worker group.
                try:
                    os.killpg(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                child = None
                if stop_requested:
                    return 130
                if returncode != NONFINITE_EXIT_CODE:
                    return returncode if returncode >= 0 else 128 - returncode
                try:
                    report = _read_report(report_path, child_pid)
                except (OSError, ValueError, TypeError) as error:
                    print(f"[recovery] Stopped: {error}", flush=True)
                    return NONFINITE_EXIT_CODE
                checkpoint = report["checkpoint"]
                permitted = budget.allow(checkpoint["iteration"])
                report["recovery_action"] = (
                    "restart" if permitted else "stop_no_progress"
                )
                report["restart_number"] = budget.restarts
                report["time_unix"] = time.time()
                history = Path(report["experiment_dir"]) / "training_recovery.jsonl"
                with history.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report) + "\n")
                if not permitted:
                    print(
                        f"[recovery] Stopped after {max_restarts} restarts without a newer "
                        f"regular checkpoint (anchor {budget.checkpoint_iter:,}).",
                        flush=True,
                    )
                    return NONFINITE_EXIT_CODE
                next_state = checkpoint["state_path"]
                seed = report["manual_seed"]
                print(
                    f"[recovery] {report['candidate_iter']:,} -> {checkpoint['iteration']:,}; "
                    f"reloading {next_state} in {restart_delay:g}s "
                    f"(recovery {budget.restarts}/{max_restarts} for this checkpoint).",
                    flush=True,
                )
                time.sleep(restart_delay)
            return 130
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
        for sig, handler in previous.items():
            signal.signal(sig, handler)
