"""Temporarily profile real updates executed by the admitted overlap trainer."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import torch

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system import engine_overlap
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json

PROFILE_WAIT = 20
PROFILE_WARMUP = 2
PROFILE_ACTIVE = 5
PROFILE_UPDATES = PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE


class ProfileAverages(Protocol):
    """Aggregate table surface used by the trace publisher."""

    def table(self, *, sort_by: str, row_limit: int) -> str: ...


class ProfilerLike(Protocol):
    """Minimal profiler surface kept injectable for CPU-only contract tests."""

    def start(self) -> None: ...

    def step(self) -> None: ...

    def stop(self) -> None: ...

    def export_chrome_trace(self, path: str) -> None: ...

    def key_averages(self) -> ProfileAverages: ...


def _relative_trace_path(path: Path) -> Path:
    """Return an ASCII artifact path for Kineto while retaining the repository cwd."""
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError("Profiled TRAIN40 execution requires the repository as cwd")
    relative = path.resolve().relative_to(ROOT.resolve())
    try:
        str(relative).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Kineto trace path must be ASCII relative to the repository") from exc
    return relative


def _profile_directory(output: Path, arm: str) -> Path:
    directory = output / "fits" / f"{arm}_seed7" / "full_step_profile"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _trace_callback(
    output: Path, directory: Path, commit_observation: dict[str, Any]
) -> Callable[[Any], None]:
    """Export aggregate tables and a trace without serializing tensor payloads."""
    trace = directory / "FULL_STEP_TRACE.json"
    cpu = directory / "FULL_STEP_CPU.txt"
    cuda = directory / "FULL_STEP_CUDA.txt"

    def publish(profiler: ProfilerLike) -> None:
        profiler.export_chrome_trace(str(_relative_trace_path(trace)))
        averages = profiler.key_averages()
        cpu.write_text(
            averages.table(sort_by="self_cpu_time_total", row_limit=100), encoding="utf-8"
        )
        cuda.write_text(
            averages.table(sort_by="self_cuda_time_total", row_limit=100), encoding="utf-8"
        )
        artifacts = [trace, cpu, cuda]
        atomic_json(
            directory / "FULL_STEP_PROFILE_MANIFEST.json",
            {
                "schema": "train40_full_step_profile_v1",
                "schedule": {
                    "wait": PROFILE_WAIT,
                    "warmup": PROFILE_WARMUP,
                    "active": PROFILE_ACTIVE,
                    "repeat": 1,
                },
                "scheduled_scientific_updates": PROFILE_UPDATES,
                "commit_observation": commit_observation,
                "trace_published_utc": datetime.now(UTC).isoformat(),
                "additional_optimizer_updates": 0,
                "contains_raw_tensor_payloads": False,
                "artifacts": [
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": digest(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in artifacts
                ],
            },
        )

    return publish


class CommitProfile:
    """Advance a bounded profiler after original durable scientific commits."""

    def __init__(
        self,
        profiler: ProfilerLike,
        directory: Path,
        commit_observation: dict[str, Any] | None = None,
    ) -> None:
        self.profiler = profiler
        self.directory = directory
        self.commits = 0
        self.started = False
        self.finished = False
        self.error: str | None = None
        self.commit_observation = commit_observation if commit_observation is not None else {}

    def start(self) -> None:
        self.profiler.start()
        self.started = True

    def after_commit(self, committed: int | None = None) -> None:
        """Count one already committed update and disable profiling at the fixed boundary."""
        if self.finished:
            return
        self.commits += 1
        if committed is not None:
            values = self.commit_observation.setdefault("scheduled_commits", [])
            values.append(int(committed))
            self.commit_observation["checkpoint_start_committed"] = values[0] - 1
            self.commit_observation["scheduled_commit_range"] = [values[0], values[-1]]
            if len(values) >= PROFILE_UPDATES:
                self.commit_observation["active_trace_commit_range"] = [
                    values[PROFILE_WAIT + PROFILE_WARMUP],
                    values[PROFILE_UPDATES - 1],
                ]
        try:
            self.profiler.step()
            if self.commits >= PROFILE_UPDATES:
                self.finish(suppress_errors=True)
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.finish(suppress_errors=True)

    def finish(self, *, suppress_errors: bool = False) -> None:
        """Stop at most once; cleanup errors never mask the admitted trainer."""
        if self.finished:
            return
        self.finished = True
        if self.started:
            try:
                self.profiler.stop()
            except BaseException as exc:
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"
                if not suppress_errors:
                    raise
        if self.error is not None:
            atomic_json(
                self.directory / "FULL_STEP_PROFILE_FAILURE.json",
                {
                    "error": self.error,
                    "committed_scientific_updates_observed": self.commits,
                    "additional_optimizer_updates": 0,
                    "training_continues_without_profiler": True,
                },
            )


def _verify_execution(output: Path) -> dict:
    """Verify both published baselines and this wrapper before CUDA profiling starts."""
    coordination_path = output / "COORDINATION_FREEZE.json"
    pause_path = output / "CONTROLLER_PAUSE_SAFE_FREEZE.json"
    execution_path = output / "EXECUTION_PROFILE_FREEZE.json"
    coordination, pause, execution = map(read, (coordination_path, pause_path, execution_path))
    for freeze in (coordination, pause, execution):
        verify_sources(freeze)
    if execution["coordination_freeze_sha256"] != digest(coordination_path):
        raise ValueError("Execution profile does not bind the coordination freeze")
    if execution["controller_pause_safe_freeze_sha256"] != digest(pause_path):
        raise ValueError("Execution profile does not bind the pause-safe freeze")
    if execution["engine_profiled_sha256"] != digest(Path(__file__)):
        raise ValueError("Profiled engine differs from its execution freeze")
    return execution


def run(output: Path, arm: str) -> None:
    """Run the unchanged overlap trainer with profiling only for its first 27 commits."""
    _verify_execution(output)
    directory = _profile_directory(output, arm)
    execution_path = output / "EXECUTION_PROFILE_FREEZE.json"
    atomic_json(
        directory / "PROFILE_WRAPPER_RUNTIME.json",
        {
            "schema": "train40_profile_wrapper_runtime_v1",
            "execution_profile_freeze_sha256": digest(execution_path),
            "overlap_engine_sha256": digest(Path(engine_overlap.__file__)),
            "profiled_engine_sha256": digest(Path(__file__)),
            "checkpoint_contract_unchanged": True,
            "additional_optimizer_updates": 0,
            "profile_steps_are_actual_scientific_commits": True,
        },
    )
    commit_observation: dict[str, Any] = {
        "wrapper_started_utc": datetime.now(UTC).isoformat(),
        "scheduled_commits": [],
    }
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=PROFILE_WAIT, warmup=PROFILE_WARMUP, active=PROFILE_ACTIVE, repeat=1
        ),
        on_trace_ready=_trace_callback(output, directory, commit_observation),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    session = CommitProfile(profiler, directory, commit_observation)
    original_commit = DurableState.commit_update

    def commit_update(self: DurableState) -> None:
        original_commit(self)
        session.after_commit(self.committed)
        if session.finished:
            DurableState.commit_update = original_commit

    DurableState.commit_update = commit_update
    try:
        session.start()
        engine_overlap.run(output, arm)
    finally:
        DurableState.commit_update = original_commit
        session.finish(suppress_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)
