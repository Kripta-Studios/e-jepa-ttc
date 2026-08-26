# ruff: noqa: E501
"""Executable, fail-closed job planning for Scientific Recovery V8.

The module deliberately plans and records work; it never manufactures a model
metric or treats a planned command as a completed experiment. It is shared by
the small stage CLIs and the PowerShell scheduler so their resume and sealed
split rules cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from csv import DictReader
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.scientific_provenance import (
    refuse_scientific_bypass_env,
    require_clean_scientific_worktree,
)

ROOT = Path(__file__).resolve().parents[3]
SEALED_MARKERS = ("public_validation", "private_test", "evttc_test", "codabench")


class V8JobIntegrityError(RuntimeError):
    """Raised when a planned V8 job could violate its frozen protocol."""


@dataclass(frozen=True)
class V8JobOutcome:
    """Resume decision for one run directory."""

    status: str
    resume: bool
    detail: str


@dataclass(frozen=True)
class V8Job:
    """A command tied to one frozen fold configuration and immutable hashes."""

    name: str
    config_path: Path
    config_sha256: str
    model_path: Path | None
    model_sha256: str | None
    output_dir: Path
    command: tuple[str, ...]
    outcome: V8JobOutcome


def sha256_file(path: Path) -> str:
    """Hash an exact file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_signed_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Sign and atomically publish an honest job-state artifact."""

    if payload.get("status") in {"completed", "complete", "passed", "success"}:
        raise V8JobIntegrityError("job substrate may not publish completed result claims")
    sign_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return payload


def _assert_not_sealed(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered.endswith("_opened") or lowered.endswith("_used_for_selection"):
                if nested is not False:
                    raise V8JobIntegrityError(f"sealed split flag is not false in {label}: {key}")
            _assert_not_sealed(nested, label=label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_not_sealed(nested, label=label)
    elif isinstance(value, str):
        normalized = value.lower().replace("\\", "/")
        if any(marker in normalized for marker in SEALED_MARKERS):
            raise V8JobIntegrityError(f"sealed evaluation source rejected in {label}: {value}")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise V8JobIntegrityError(f"invalid V8 job YAML: {path}") from error
    if not isinstance(value, dict):
        raise V8JobIntegrityError(f"V8 job config must be a mapping: {path}")
    _assert_not_sealed(value, label=str(path))
    return value


def _model_path(config_path: Path, config: Mapping[str, Any]) -> Path | None:
    raw = config.get("model_config")
    if not isinstance(raw, str):
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.is_file():
        raise V8JobIntegrityError(f"frozen model config is missing: {candidate}")
    return candidate.resolve()


def assess_v8_job(run_dir: Path) -> V8JobOutcome:
    """Return new/resume/completed while preserving corrupt checkpoints intact."""

    summary = run_dir / "summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise V8JobIntegrityError(f"invalid run summary: {summary}") from error
        if not isinstance(payload, dict) or not verify_artifact_hash(payload):
            raise V8JobIntegrityError(f"invalid signed run summary: {summary}")
        _assert_not_sealed(payload, label=str(summary))
        complete_statuses = {
            "completed",
            "complete",
            "passed",
            "success",
            "completed_train_only_grouped_dev",
        }
        if str(payload.get("status", "")).lower() not in complete_statuses:
            raise V8JobIntegrityError(f"run summary is not completed: {summary}")
        return V8JobOutcome("completed", False, "signed completed run summary")
    checkpoint = run_dir / "state" / "last.pt"
    if not checkpoint.exists():
        return V8JobOutcome("new", False, "no completed summary or state/last.pt")
    try:
        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint is not a mapping")
    except Exception as error:  # torch's exact exception is implementation dependent.
        failure = {
            "artifact_type": "scientific_recovery_v8_job_failure_v1",
            "status": "failed_integrity",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": str(checkpoint),
            "detail": f"corrupt state/last.pt: {type(error).__name__}: {error}",
        }
        _atomic_signed_json(run_dir / "failed_integrity.json", failure)
        return V8JobOutcome("failed_integrity", False, failure["detail"])
    return V8JobOutcome("resume", True, "valid state/last.pt without completed summary")


def build_fold_jobs(
    *,
    configs: Sequence[Path],
    output_root: Path,
    device: str,
    max_parallel: int = 1,
    allowed_seeds: tuple[int, ...] = (7,),
) -> tuple[V8Job, ...]:
    """Build exactly one real supervised training command per frozen fold."""

    if max_parallel < 1:
        raise ValueError("max_parallel must be at least one")
    jobs: list[V8Job] = []
    seen_identities: set[tuple[str, int, int]] = set()
    for source in sorted((Path(path).resolve() for path in configs), key=lambda item: item.name):
        if not source.is_file():
            raise V8JobIntegrityError(f"frozen fold config is missing: {source}")
        config = _load_yaml(source)
        experiment = config.get("experiment")
        data = config.get("data")
        if not isinstance(experiment, Mapping) or not isinstance(data, Mapping):
            raise V8JobIntegrityError(f"V8 fold config lacks experiment/data: {source}")
        name = str(experiment.get("name", ""))
        arm = str(experiment.get("arm", ""))
        seed = experiment.get("seed")
        fold = data.get("outer_fold")
        if not name or seed not in allowed_seeds or not arm or not isinstance(fold, int):
            raise V8JobIntegrityError(
                f"V8 fold config seed/identity is outside this frozen job plan: {source}"
            )
        identity = (arm, int(seed), fold)
        if identity in seen_identities:
            raise V8JobIntegrityError(f"duplicate seed/fold in V8 job plan: {identity}")
        seen_identities.add(identity)
        if data.get("opened_splits") != ["train"]:
            raise V8JobIntegrityError(f"V8 job must only open train data: {source}")
        # Canonical run directories are arm/fold/seed based so producers and
        # aggregators share one immutable path contract independent of display names.
        run_dir = output_root.resolve() / f"{arm}_fold{fold}_seed{seed}"
        outcome = assess_v8_job(run_dir)
        command = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/train_scientific_recovery_v8_temporal.py",
            "--config",
            str(source),
            "--output-dir",
            str(run_dir),
            "--device",
            device,
        ]
        if outcome.resume:
            command.append("--resume")
        model_path = _model_path(source, config)
        jobs.append(
            V8Job(
                name=name,
                config_path=source,
                config_sha256=sha256_file(source),
                model_path=model_path,
                model_sha256=sha256_file(model_path) if model_path else None,
                output_dir=run_dir,
                command=tuple(command),
                outcome=outcome,
            )
        )
    return tuple(jobs)


def write_v8_job_state(
    *,
    job: V8Job,
    status: str,
    protocol_hash: str,
    manifest_hash: str,
    detail: str = "job planned",
) -> dict[str, Any]:
    """Record a signed non-result job transition; completion belongs to the trainer."""

    if status not in {"planned", "running", "failed", "failed_integrity", "skipped"}:
        raise ValueError("job state must be planned/running/failed/failed_integrity/skipped")
    if len(protocol_hash) != 64 or len(manifest_hash) != 64:
        raise ValueError("protocol_hash and manifest_hash must be SHA-256 values")
    return _atomic_signed_json(
        job.output_dir / "job_state.json",
        {
            "artifact_type": "scientific_recovery_v8_job_state_v1",
            "status": status,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "name": job.name,
            "detail": detail,
            "config": {"path": str(job.config_path), "sha256": job.config_sha256},
            "model_config": (
                {"path": str(job.model_path), "sha256": job.model_sha256}
                if job.model_path is not None
                else None
            ),
            "protocol_artifact_sha256": protocol_hash,
            "frozen_manifest_artifact_sha256": manifest_hash,
            "command": list(job.command),
            "resume_state": job.outcome.status,
            "closed_evaluation": {
                "public_validation_used_for_selection": False,
                "private_test_opened": False,
                "evttc_test_opened": False,
                "codabench_opened": False,
            },
        },
    )


def execute_jobs(
    jobs: Sequence[V8Job],
    *,
    protocol_hash: str,
    manifest_hash: str,
    dry_run: bool,
    max_parallel: int = 1,
    command_runner: Callable[[V8Job], int] | None = None,
) -> list[dict[str, Any]]:
    """Execute bounded concurrent commands, without treating plans as results."""

    refuse_scientific_bypass_env()
    require_clean_scientific_worktree()
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least one")

    def invoke(job: V8Job) -> int:
        if command_runner is not None:
            return int(command_runner(job))
        job.output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = job.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        command_path = log_dir / "command.json"
        command_path.write_text(
            json.dumps({"command": list(job.command), "cwd": str(ROOT)}, indent=2) + "\n",
            encoding="utf-8",
        )
        # Append on resume so the complete process history is retained.  Each
        # invocation is bracketed with an ISO timestamp in both streams.
        started = datetime.now(UTC).isoformat()
        with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout_handle, \
             stderr_path.open("a", encoding="utf-8", buffering=1) as stderr_handle:
            stdout_handle.write(f"\n===== V8 JOB START {started} =====\n")
            stdout_handle.write("COMMAND: " + " ".join(job.command) + "\n")
            stderr_handle.write(f"\n===== V8 JOB START {started} =====\n")
            result = subprocess.run(
                list(job.command),
                cwd=ROOT,
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            finished = datetime.now(UTC).isoformat()
            stdout_handle.write(
                f"===== V8 JOB END {finished} exit={result.returncode} =====\n"
            )
            stderr_handle.write(
                f"===== V8 JOB END {finished} exit={result.returncode} =====\n"
            )
        return int(result.returncode)

    outputs: dict[str, dict[str, Any]] = {}
    queued: list[V8Job] = []
    for job in jobs:
        if job.outcome.status == "completed":
            outputs[job.name] = {
                "name": job.name,
                "status": "reused_completed",
                "command": list(job.command),
            }
        elif job.outcome.status == "failed_integrity":
            raise V8JobIntegrityError(
                f"resume checkpoint is corrupt for {job.name}: {job.outcome.detail}"
            )
        elif dry_run:
            outputs[job.name] = {
                "name": job.name,
                "status": "planned",
                "command": list(job.command),
            }
        else:
            queued.append(job)
    if dry_run:
        return [outputs[job.name] for job in jobs]

    failure: V8JobIntegrityError | None = None
    running: dict[Future[int], V8Job] = {}
    iterator = iter(queued)
    with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="v8-job") as executor:
        while failure is None and len(running) < max_parallel:
            try:
                job = next(iterator)
            except StopIteration:
                break
            write_v8_job_state(
                job=job,
                status="running",
                protocol_hash=protocol_hash,
                manifest_hash=manifest_hash,
            )
            running[executor.submit(invoke, job)] = job
        while running:
            completed, _ = wait(tuple(running), return_when="FIRST_COMPLETED")
            for future in completed:
                job = running.pop(future)
                try:
                    exit_code = future.result()
                    detail = f"trainer exited {exit_code}"
                except Exception as error:
                    exit_code = -1
                    detail = f"trainer raised {type(error).__name__}: {error}"
                if exit_code == 0 and assess_v8_job(job.output_dir).status == "completed":
                    outputs[job.name] = {
                        "name": job.name,
                        "status": "completed",
                        "command": list(job.command),
                    }
                    continue
                write_v8_job_state(
                    job=job,
                    status="failed",
                    protocol_hash=protocol_hash,
                    manifest_hash=manifest_hash,
                    detail=(
                        detail
                        if exit_code != 0
                        else "trainer exited zero without a signed completed summary"
                    ),
                )
                failure = V8JobIntegrityError(f"training failed for {job.name}: {detail}")
            while failure is None and len(running) < max_parallel:
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                write_v8_job_state(
                    job=job,
                    status="running",
                    protocol_hash=protocol_hash,
                    manifest_hash=manifest_hash,
                )
                running[executor.submit(invoke, job)] = job
    if failure is not None:
        raise failure
    return [outputs[job.name] for job in jobs]


def derive_cache_selection(
    protocol: Mapping[str, Any], *, expected_rows: int = 8192, root: Path | None = None
) -> list[dict[str, str]]:
    """Derive exact selection metadata from paired signed train-only A5/Garl OOF CSVs."""

    base = (root or ROOT).resolve()
    sources = protocol.get("sources")
    if not isinstance(sources, Mapping):
        raise V8JobIntegrityError("frozen protocol lacks paired OOF sources")

    def resolve(key: str) -> Path:
        source = sources.get(key)
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            raise V8JobIntegrityError(f"frozen protocol lacks sources.{key}.path")
        raw = str(source["path"])
        _assert_not_sealed(raw, label=f"sources.{key}")
        path = Path(raw)
        path = path if path.is_absolute() else base / path
        if not path.is_file():
            raise V8JobIntegrityError(f"signed train-only OOF source is missing: {path}")
        return path

    def read(path: Path) -> dict[str, dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(DictReader(handle))
        required = {"sample_token", "sequence_id", "track_id", "target_ttc_s", "fold"}
        if not records or not required.issubset(records[0]):
            raise V8JobIntegrityError(f"OOF source lacks V8 identity columns: {path}")
        values = {str(row["sample_token"]): dict(row) for row in records}
        if len(values) != len(records):
            raise V8JobIntegrityError(f"OOF source has duplicate sample tokens: {path}")
        return values

    a5, garl = read(resolve("a5_oof_predictions")), read(resolve("garl_oof_predictions"))
    if set(a5) != set(garl):
        raise V8JobIntegrityError("paired A5/Garl OOF sources have different token universes")
    selected: list[dict[str, str]] = []
    for token in sorted(a5):
        left, right = a5[token], garl[token]
        identity = ("sequence_id", "track_id", "target_ttc_s", "fold")
        if any(left[key] != right[key] for key in identity):
            raise V8JobIntegrityError(f"paired OOF identity mismatch for {token}")
        selected.append(
            {
                "sample_token": token,
                "sequence_id": left["sequence_id"],
                "track_id": left["track_id"],
                "target_ttc": left["target_ttc_s"],
                "outer_fold": left["fold"],
            }
        )
    if len(selected) != expected_rows:
        raise V8JobIntegrityError(
            f"paired OOF selection has {len(selected)} rows; expected {expected_rows}"
        )
    return selected


def signed_derived_multiseed_manifest(
    *, candidate: str, source: Mapping[str, Any], output_path: Path
) -> dict[str, Any]:
    """Freeze only seeds 13/23 mechanically from one signed candidate nomination."""

    if candidate != str(source.get("candidate_id", source.get("arm", ""))):
        raise V8JobIntegrityError("derived multiseed candidate differs from signed nomination")
    nominated = source.get("multiseed_replication_candidate") is True
    gate = source.get("gate_decision", source.get("gates", {}))
    if not nominated and (
        not isinstance(gate, Mapping) or gate.get("multiseed_replication_candidate") is not True
    ):
        raise V8JobIntegrityError("candidate has not been nominated for multiseed replication")
    _assert_not_sealed(source, label="seed-7 nomination")
    payload: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_multiseed_derived_manifest_v1",
        "status": "frozen_before_multiseed_replication",
        "candidate_id": candidate,
        "seeds": [13, 23],
        "source_seed": 7,
        "source_artifact_sha256": source.get("artifact_sha256"),
        "no_tuning": True,
        "no_reselection": True,
        "closed_evaluation": {
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
    }
    sign_artifact(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return payload


def clone_multiseed_configs(
    *,
    candidate: str,
    source_configs: Sequence[Path],
    output_dir: Path,
) -> list[Path]:
    """Mechanically clone candidate configs for seeds 13/23 without retuning."""

    if not source_configs:
        raise V8JobIntegrityError("cannot clone multiseed configs without frozen source configs")
    generated: list[Path] = []
    for original in sorted(
        (Path(value).resolve() for value in source_configs), key=lambda value: value.name
    ):
        config = _load_yaml(original)
        experiment = config.get("experiment")
        if not isinstance(experiment, dict) or str(experiment.get("arm")) != candidate:
            raise V8JobIntegrityError(
                f"source config does not belong to nominated candidate: {original}"
            )
        if experiment.get("seed") != 7:
            raise V8JobIntegrityError(f"multiseed source config must be frozen seed 7: {original}")
        for seed in (13, 23):
            cloned = json.loads(json.dumps(config))
            cloned_experiment = cloned["experiment"]
            cloned_experiment["seed"] = seed
            cloned_experiment["name"] = str(experiment["name"]).replace("seed7", f"seed{seed}")
            cloned_training = cloned.get("training")
            if isinstance(cloned_training, dict) and "seed" in cloned_training:
                cloned_training["seed"] = seed
            cloned_experiment["multiseed_replication"] = {
                "source_seed": 7,
                "seeds_are_optimization_stability_replication": True,
                "external_confirmation": False,
                "no_tuning": True,
                "no_reselection": True,
            }
            target = output_dir / f"{original.stem.replace('seed7', f'seed{seed}')}.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_text(
                yaml.safe_dump(cloned, sort_keys=False), encoding="utf-8", newline="\n"
            )
            os.replace(temporary, target)
            generated.append(target)
    return generated


def plan_to_json(jobs: Sequence[V8Job]) -> dict[str, Any]:
    """Serialize commands and hashes for dry-run inspection, without result claims."""

    return {
        "status": "planned",
        "max_parallel_supported": True,
        "jobs": [
            {
                "name": job.name,
                "config_sha256": job.config_sha256,
                "model_sha256": job.model_sha256,
                "resume_state": job.outcome.status,
                "command": list(job.command),
            }
            for job in jobs
        ],
    }


__all__ = [
    "ROOT",
    "V8Job",
    "V8JobIntegrityError",
    "V8JobOutcome",
    "assess_v8_job",
    "build_fold_jobs",
    "clone_multiseed_configs",
    "derive_cache_selection",
    "execute_jobs",
    "plan_to_json",
    "sha256_file",
    "signed_derived_multiseed_manifest",
    "write_v8_job_state",
]
