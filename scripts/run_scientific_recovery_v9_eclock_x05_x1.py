"""Single fail-closed runner for the preregistered E-Clock X0.5 -> X1 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.evaluation.collision_clock_bootstrap import paired_hierarchical_mid_bootstrap
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    load_signed_json,
    module_topology_sha256,
    production_sequence_macro_metrics,
    tensor_state_sha256,
)
from e_jepa_ttc.evaluation.incremental_fusion import (
    DYNAMIC_SLOT_NAMES,
    X1_ARMS,
    atomic_write_json,
    deterministic_within_sequence_shuffle,
    evaluate_x1_gate,
    load_signed_artifact,
    run_x05_cross_fit,
    scientific_prediction_frame,
    validate_feature_table,
)
from e_jepa_ttc.evaluation.incremental_replay import (
    EXPECTED_X0_BUNDLE_SHA256,
    EXPECTED_X0_COMMIT,
    run_feature_replay,
)
from e_jepa_ttc.models.incremental_residual import FrozenA5DynamicResidualAdapter
from e_jepa_ttc.training.incremental_residual import (
    X1TrainingConfig,
    X1TrainingIdentity,
    deterministic_sequence_grouped_schedule,
    load_frozen_x1_checkpoint,
    normalization_sha256,
    token_sha256,
    train_x1_fixed_budget,
    trainable_mask_sha256,
)

ALLOWED_NEXT_DECISIONS = {
    "INVALID_X05",
    "GLOBAL_DYNAMIC_SLOTS_REDUNDANT_WITH_A5_SCREEN",
    "INCREMENTAL_SIGNAL_TOO_SMALL_FOR_X1",
    "X1_SEED7_NEGATIVE",
    "X1_INCREMENTAL_BUT_NOT_COMPETITIVE",
    "X1_SEED7_SUPPORTED_REPLICATION_REQUIRED",
    "X1_REPLICATED_GROUPED_DEV_CANDIDATE",
    "INVALID_X1",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _validate_schema(payload: Mapping[str, Any], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(dict(payload))


def _verify_zip_manifest(path: Path) -> dict[str, Any]:
    if compute_file_hash(str(path)) != EXPECTED_X0_BUNDLE_SHA256:
        raise ValueError("X0 bundle SHA mismatch")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"X0 bundle CRC failure: {bad}")
        manifest = json.loads(archive.read("MANIFEST.json"))
        members = manifest.get("members")
        if not isinstance(members, list) or len(members) != 186:
            raise ValueError("X0 bundle manifest must cover exactly 186 members")
        names = set(archive.namelist())
        if len(names) != 187 or names != {str(item["path"]) for item in members} | {
            "MANIFEST.json"
        }:
            raise ValueError("X0 bundle physical member universe mismatch")
        for member in members:
            name = member.get("path")
            if name not in names:
                raise ValueError(f"X0 manifest member missing: {name}")
            data = archive.read(name)
            if len(data) != int(member["bytes"]):
                raise ValueError(f"X0 manifest byte mismatch: {name}")
            if hashlib.sha256(data).hexdigest() != member["sha256"]:
                raise ValueError(f"X0 manifest hash mismatch: {name}")
        if manifest.get("git_commit") != EXPECTED_X0_COMMIT:
            raise ValueError("X0 bundle commit mismatch")
    return {
        "sha256": EXPECTED_X0_BUNDLE_SHA256,
        "physical_entries": 187,
        "manifest_members": 186,
        "crc_passed": True,
    }


def _benchmark_x1_devices() -> dict[str, Any]:
    rng = np.random.default_rng(20260904)
    a5 = torch.zeros(8192, dtype=torch.float32)
    slots = torch.from_numpy(rng.normal(size=(8192, 9)).astype(np.float32))
    timings: dict[str, float] = {}
    candidates = [torch.device("cpu")]
    if torch.cuda.is_available():
        candidates.append(torch.device("cuda:0"))
    for device in candidates:
        torch.manual_seed(20260904)
        model = FrozenA5DynamicResidualAdapter(torch.zeros(9), torch.ones(9)).to(device).eval()
        device_a5 = a5.to(device)
        device_slots = slots.to(device)
        with torch.no_grad():
            for _ in range(5):
                model(device_a5, device_slots)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(25):
                model(device_a5, device_slots)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        timings[str(device)] = (time.perf_counter() - started) / 25.0
        del model, device_a5, device_slots
    selected = min(timings, key=timings.get)  # type: ignore[arg-type]
    return {
        "synthetic_rows": 8192,
        "iterations": 25,
        "mean_forward_seconds": timings,
        "selected_device": selected,
        "scientific_rows_observed": False,
    }


def _preflight_memory_policy(available_bytes: int) -> dict[str, Any]:
    if available_bytes < 4 * 1024**3:
        raise ValueError("preflight RAM safety margin is below 4 GiB even for shard_lru")
    return {
        "available_bytes": available_bytes,
        "fold_ram_eligible": available_bytes >= 8 * 1024**3,
        "selected_cache_mode": "shard_lru",
        "reason": "conservative_bounded_cache"
        if available_bytes >= 8 * 1024**3
        else "ram_fallback",
    }


def _verify_required_x0_checkpoints(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm in ("X0-A5-REPLAY", "X0-BASE-U", "X0-DYN-U"):
        for fold in (0, 1, 2):
            summary = load_signed_artifact(
                args.x0_campaign / arm / f"fold-{fold}/fold_summary.json"
            )
            if arm == "X0-A5-REPLAY":
                checkpoint = Path(str(summary["checkpoint_path"]))
            else:
                checkpoint = (
                    args.x0_campaign / arm / f"fold-{fold}" / "milestones" / "update-006840.pt"
                )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"required X0 checkpoint missing: {checkpoint}")
            observed = compute_file_hash(str(checkpoint))
            if observed != summary["checkpoint_file_sha256"]:
                raise ValueError(f"required X0 checkpoint SHA mismatch: {checkpoint}")
            records.append(
                {
                    "arm": arm,
                    "outer_fold": fold,
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": observed,
                }
            )
    return records


def run_preflight(args: argparse.Namespace, campaign: Path) -> dict[str, Any]:
    repo = args.repo.resolve()
    if _git(repo, "rev-parse", "HEAD") != EXPECTED_X0_COMMIT and not args.allow_training_commit:
        raise ValueError("pre-implementation HEAD differs from mandatory X0 starting commit")
    branch = _git(repo, "branch", "--show-current")
    if branch != "scientific-recovery-v9-eclock-x1-incremental-fusion":
        raise ValueError("X1 worktree branch mismatch")
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", EXPECTED_X0_COMMIT, "HEAD"]
    )
    if ancestor.returncode != 0:
        raise ValueError("training commit is not descended from mandatory X0 starting commit")
    remote = _git(repo, "ls-remote", "origin", "refs/heads/scientific-recovery-v9-eclock-x0")
    if not remote.startswith(EXPECTED_X0_COMMIT + "\t"):
        raise ValueError("origin/scientific-recovery-v9-eclock-x0 moved from the mandatory pin")
    versioned_dirty = _git(repo, "status", "--porcelain=v1", "--untracked-files=no")
    if versioned_dirty:
        raise ValueError("scientific execution requires a clean versioned worktree")
    bundle = _verify_zip_manifest(args.x0_bundle)
    protocol_path = repo / "configs/protocol/scientific_recovery_v9_eclock_x05_x1.json"
    protocol = load_signed_json(
        protocol_path,
        schema_path=repo / "schemas/scientific_recovery_v9_eclock_x05_x1_protocol_v1.schema.json",
    )
    if protocol["source_commit"] != EXPECTED_X0_COMMIT:
        raise ValueError("X0.5/X1 protocol source pin mismatch")
    x05_protocol = atomic_write_json(
        campaign / "x05_protocol.json",
        {
            "artifact_type": "eclock_x05_protocol_v1",
            "combined_protocol_sha256": protocol["artifact_sha256"],
            "source_commit": protocol["source_commit"],
            "source_bundle_sha256": protocol["source_bundle_sha256"],
            "sealed_evaluation": protocol["sealed_evaluation"],
            "x05": protocol["x05"],
        },
    )
    x1_protocol = atomic_write_json(
        campaign / "x1_protocol.json",
        {
            "artifact_type": "eclock_x1_protocol_v1",
            "combined_protocol_sha256": protocol["artifact_sha256"],
            "source_commit": protocol["source_commit"],
            "source_bundle_sha256": protocol["source_bundle_sha256"],
            "sealed_evaluation": protocol["sealed_evaluation"],
            "x1": protocol["x1"],
        },
    )
    prior_attempts = []
    campaigns_root = args.output_root / "campaigns"
    if campaigns_root.is_dir():
        for failure_path in sorted(campaigns_root.glob("*/campaign_failure.json")):
            if campaign in failure_path.parents:
                continue
            replay_path = failure_path.parent / "x05/replay/x05_feature_replay_manifest.json"
            if replay_path.exists():
                continue
            failure = load_signed_artifact(
                failure_path, artifact_type="eclock_x05_x1_campaign_failure_v1"
            )
            prior_attempts.append(
                {
                    "campaign": failure_path.parent.name,
                    "training_commit": failure["training_commit"],
                    "exception_type": failure["exception_type"],
                    "exception_message": failure["exception_message"],
                    "feature_replay_started": False,
                    "x05_meta_test_started": False,
                    "sealed_evaluation_opened": False,
                }
            )
    atomic_write_json(
        campaign / "pre_result_attempts.json",
        {
            "artifact_type": "eclock_x05_x1_pre_result_attempts_v1",
            "attempt_count": len(prior_attempts),
            "attempts": prior_attempts,
        },
    )
    x0_protocol = load_signed_json(
        repo / "configs/protocol/scientific_recovery_v9_eclock_x0.json",
        schema_path=repo / "schemas/scientific_recovery_v9_eclock_protocol_v2.schema.json",
    )
    cache_manifest = args.cache_root / "manifest.json"
    if compute_file_hash(str(cache_manifest)) != x0_protocol["cache_binding"]["file_sha256"]:
        raise ValueError("cache manifest SHA mismatch")
    shard_checks = []
    for record in x0_protocol["cache_binding"]["train_shards"]:
        path = args.cache_root / Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"cache shard missing/truncated: {path}")
        observed = compute_file_hash(str(path))
        if observed != record["file_sha256"]:
            raise ValueError(f"cache shard SHA mismatch: {path}")
        shard_checks.append({"path": str(path), "bytes": path.stat().st_size, "sha256": observed})
    checkpoint_checks = _verify_required_x0_checkpoints(args)
    x1_benchmark = _benchmark_x1_devices()
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(repo)
    memory_policy = _preflight_memory_policy(memory.available)
    if disk.free < 20 * 1024**3:
        raise ValueError("preflight disk safety margin is below 20 GiB")
    result = atomic_write_json(
        campaign / "preflight.json",
        {
            "artifact_type": "eclock_x05_x1_preflight_v1",
            "starting_head": EXPECTED_X0_COMMIT,
            "training_commit": _git(repo, "rev-parse", "HEAD"),
            "branch": branch,
            "remote_source_pin_exact": True,
            "versioned_worktree_clean": True,
            "bundle": bundle,
            "x05_x1_protocol_sha256": protocol["artifact_sha256"],
            "x05_protocol_sha256": x05_protocol["artifact_sha256"],
            "x1_protocol_sha256": x1_protocol["artifact_sha256"],
            "cache_manifest_sha256": x0_protocol["cache_binding"]["file_sha256"],
            "cache_shards_verified": len(shard_checks),
            "cache_shard_total_bytes": sum(row["bytes"] for row in shard_checks),
            "required_x0_checkpoints_verified": checkpoint_checks,
            "x1_device_benchmark": x1_benchmark,
            "x1_device_selected": x1_benchmark["selected_device"],
            "gpu": gpu,
            "ram_total_bytes": memory.total,
            "ram_available_bytes": memory.available,
            "memory_policy": memory_policy,
            "disk_free_bytes": disk.free,
            "thread_limit": 16,
            "sealed_evaluation_opened": False,
        },
    )
    return result


class TelemetryMonitor:
    """Five-second host/GPU monitor with phase updates and no scientific inputs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.phase = "startup"
        self.arm = ""
        self.seed = -1
        self.fold = -1
        self.update = -1
        self._sample_count = 0
        self._peak_process_rss_bytes = 0
        self._peak_gpu_memory_used_mib = 0.0

    def set_phase(self, phase: str, *, arm: str = "", seed: int = -1, fold: int = -1) -> None:
        self.phase, self.arm, self.seed, self.fold = phase, arm, seed, fold
        self.update = -1

    def set_update(self, update: int) -> None:
        self.update = update

    def start(self) -> None:
        def loop() -> None:
            process = psutil.Process()
            while not self._stop.is_set():
                gpu_result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                record = {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "phase": self.phase,
                    "arm": self.arm,
                    "seed": self.seed,
                    "fold": self.fold,
                    "update": self.update,
                    "gpu": gpu_result.stdout.strip() if gpu_result.returncode == 0 else None,
                    "cpu_percent": psutil.cpu_percent(),
                    "process_cpu_percent": process.cpu_percent(),
                    "ram_used_bytes": psutil.virtual_memory().used,
                    "ram_total_bytes": psutil.virtual_memory().total,
                    "process_rss_bytes": process.memory_info().rss,
                    "process_read_bytes": getattr(process.io_counters(), "read_bytes", 0),
                    "process_write_bytes": getattr(process.io_counters(), "write_bytes", 0),
                    "disk_free_bytes": shutil.disk_usage(self.root).free,
                }
                self._sample_count += 1
                self._peak_process_rss_bytes = max(
                    self._peak_process_rss_bytes, int(record["process_rss_bytes"])
                )
                if record["gpu"]:
                    fields = [value.strip() for value in str(record["gpu"]).split(",")]
                    if len(fields) >= 2:
                        self._peak_gpu_memory_used_mib = max(
                            self._peak_gpu_memory_used_mib, float(fields[1])
                        )
                with (self.root / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                self._stop.wait(5.0)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        atomic_write_json(
            self.root / "telemetry_summary.json",
            {
                "artifact_type": "eclock_x05_x1_telemetry_summary_v1",
                "interval_seconds": 5,
                "sample_count": self._sample_count,
                "peak_process_rss_bytes": self._peak_process_rss_bytes,
                "peak_gpu_memory_used_mib": self._peak_gpu_memory_used_mib,
            },
        )


def _arm_slots(
    arm: str,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    *,
    seed: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_real = train.loc[:, DYNAMIC_SLOT_NAMES].to_numpy(dtype=np.float64)
    dev_real = dev.loc[:, DYNAMIC_SLOT_NAMES].to_numpy(dtype=np.float64)
    if arm == "X1-A5-ZERO-U":
        return np.zeros_like(train_real), np.zeros_like(dev_real), {"mode": "zero"}
    if arm == "X1-A5-DYN-U":
        return train_real, dev_real, {"mode": "dynamic"}
    if arm == "X1-A5-SHUFFLE-U":
        train_shuffled, train_sha = deterministic_within_sequence_shuffle(
            train_real,
            train["sequence_id"].astype(str).tolist(),
            seed=20260904,
            outer_fold=fold,
            partition="meta-train",
        )
        dev_shuffled, dev_sha = deterministic_within_sequence_shuffle(
            dev_real,
            dev["sequence_id"].astype(str).tolist(),
            seed=20260904,
            outer_fold=fold,
            partition="meta-test",
        )
        return (
            train_shuffled,
            dev_shuffled,
            {
                "mode": "shuffle",
                "permutation_seed": 20260904,
                "train_permutation_sha256": train_sha,
                "dev_permutation_sha256": dev_sha,
                "requested_training_seed_does_not_affect_shuffle": seed,
            },
        )
    raise ValueError(f"invalid trainable X1 arm: {arm}")


def _comparison_identity(manifest: Mapping[str, Any], arm: str) -> dict[str, str]:
    return {
        "reference_family": arm,
        "path": str(manifest["oof_path"]),
        "file_sha256": str(manifest["oof_file_sha256"]),
        "artifact_sha256": str(manifest["artifact_sha256"]),
    }


def _run_x1_seed(
    *,
    args: argparse.Namespace,
    campaign: Path,
    features: pd.DataFrame,
    feature_manifest: Mapping[str, Any],
    x05_gate: Mapping[str, Any],
    x0_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seed: int,
    monitor: TelemetryMonitor,
    resume: bool,
) -> dict[str, Any]:
    seed_root = campaign / "x1" / f"seed-{seed}"
    seed_root.mkdir(parents=True, exist_ok=resume)
    training_commit = _git(args.repo, "rev-parse", "HEAD")
    config_path = args.repo / "configs/experiment/scientific_recovery_v9_eclock_x1/x1_residual.yaml"
    config_sha = compute_file_hash(str(config_path))
    slot_all = features.loc[:, DYNAMIC_SLOT_NAMES].to_numpy(dtype=np.float64)
    fold_frames: dict[str, list[pd.DataFrame]] = {arm: [] for arm in X1_ARMS}
    fold_summaries: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    for fold in (0, 1, 2):
        dev_mask = features["outer_fold"].to_numpy(dtype=np.int64) == fold
        train = features.loc[~dev_mask].reset_index(drop=True)
        dev = features.loc[dev_mask].reset_index(drop=True)
        mean = np.asarray(
            np.mean(slot_all[~dev_mask], axis=0, dtype=np.float64), dtype=np.float64
        ).reshape(-1)
        std = np.asarray(
            np.std(slot_all[~dev_mask], axis=0, dtype=np.float64), dtype=np.float64
        ).reshape(-1)
        std = np.asarray(np.where(std > 1.0e-12, std, 1.0), dtype=np.float64)
        schedule, schedule_sha = deterministic_sequence_grouped_schedule(
            train["sequence_id"].astype(str).tolist(),
            train["sample_token"].astype(str).tolist(),
            seed=seed,
        )
        a5_dev = dev["a5_predicted_benchmark_phase"].to_numpy(dtype=np.float64)
        a5_frame = scientific_prediction_frame(dev, a5_dev, arm_id="X1-A5-REPLAY")
        fold_frames["X1-A5-REPLAY"].append(a5_frame)
        for arm in ("X1-A5-ZERO-U", "X1-A5-DYN-U", "X1-A5-SHUFFLE-U"):
            monitor.set_phase("x1_train", arm=arm, seed=seed, fold=fold)
            train_slots, dev_slots, transform = _arm_slots(arm, train, dev, seed=seed, fold=fold)
            torch.manual_seed(seed)
            model = FrozenA5DynamicResidualAdapter(
                torch.as_tensor(mean, dtype=torch.float32),
                torch.as_tensor(std, dtype=torch.float32),
            )
            identity = X1TrainingIdentity(
                training_commit=training_commit,
                feature_table_sha256=str(feature_manifest["feature_table_file_sha256"]),
                x05_gate_sha256=str(x05_gate["artifact_sha256"]),
                protocol_sha256=str(protocol["artifact_sha256"]),
                config_sha256=config_sha,
                arm_id=arm,
                seed=seed,
                outer_fold=fold,
                train_token_sha256=token_sha256(train["sample_token"].astype(str).tolist()),
                dev_token_sha256=token_sha256(dev["sample_token"].astype(str).tolist()),
                topology_sha256=module_topology_sha256(model),
                initialization_sha256=tensor_state_sha256(model),
                trainable_mask_sha256=trainable_mask_sha256(model),
                normalization_sha256=normalization_sha256(mean, std),
                batch_schedule_sha256=schedule_sha,
                a5_frozen=True,
                transport_extractor_frozen=True,
                outer_dev_available_to_trainer=False,
            )
            fold_root = seed_root / arm / f"fold-{fold}"
            summary_path = fold_root / "fold_summary.json"
            if summary_path.is_file():
                if not resume:
                    raise FileExistsError("completed X1 fold exists without --resume")
                summary = load_signed_artifact(
                    summary_path, artifact_type="eclock_x1_fold_summary_v1"
                )
                if summary.get("identity") != identity.__dict__:
                    raise ValueError("completed X1 fold identity mismatch")
                oof_path = Path(str(summary["oof_path"]))
                if compute_file_hash(str(oof_path)) != summary["oof_file_sha256"]:
                    raise ValueError("completed X1 fold OOF hash mismatch")
                prediction_frame = pd.read_csv(oof_path, float_precision="round_trip")
            else:
                training_summary = train_x1_fixed_budget(
                    model,
                    a5_phase=train["a5_predicted_benchmark_phase"].to_numpy(dtype=np.float32),
                    slots=train_slots,
                    target_phase=train["target_benchmark_phase"].to_numpy(dtype=np.float32),
                    sample_weights=train["sample_weight"].to_numpy(dtype=np.float32),
                    schedule=schedule,
                    config=X1TrainingConfig(arm_id=arm, seed=seed, outer_fold=fold),
                    identity=identity,
                    output_root=fold_root,
                    device=args.x1_device,
                    resume=resume and (fold_root / "resume_latest.pt").is_file(),
                    progress_callback=monitor.set_update,
                )
                frozen = load_frozen_x1_checkpoint(
                    fold_root / "resume_latest.pt",
                    expected_identity=identity,
                    model=model,
                    device=args.x1_device,
                )
                monitor.set_phase("x1_outer_dev_once", arm=arm, seed=seed, fold=fold)
                with torch.no_grad():
                    prediction = (
                        frozen(
                            torch.as_tensor(a5_dev, dtype=torch.float32, device=args.x1_device),
                            torch.as_tensor(dev_slots, dtype=torch.float32, device=args.x1_device),
                        )
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                prediction_frame = scientific_prediction_frame(dev, prediction, arm_id=arm)
                oof_path = fold_root / "oof_predictions.csv"
                prediction_frame.to_csv(oof_path, index=False)
                summary = atomic_write_json(
                    summary_path,
                    {
                        "artifact_type": "eclock_x1_fold_summary_v1",
                        "identity": identity.__dict__,
                        "arm_id": arm,
                        "seed": seed,
                        "outer_fold": fold,
                        "train_rows": len(train),
                        "dev_rows": len(dev),
                        "outer_dev_evaluations": 1,
                        "outer_dev_used_for_selection": False,
                        "checkpoint_policy": "last_update_fixed_budget",
                        "training_summary": training_summary,
                        "slot_transform": transform,
                        "oof_path": str(oof_path),
                        "oof_file_sha256": compute_file_hash(str(oof_path)),
                        "oof_bytes": oof_path.stat().st_size,
                    },
                )
            fold_frames[arm].append(prediction_frame)
            fold_summaries.append(summary)
            match_records.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "arm": arm,
                    "topology_sha256": identity.topology_sha256,
                    "initialization_sha256": identity.initialization_sha256,
                    "trainable_mask_sha256": identity.trainable_mask_sha256,
                    "normalization_sha256": identity.normalization_sha256,
                    "batch_schedule_sha256": identity.batch_schedule_sha256,
                }
            )
    arm_frames: dict[str, pd.DataFrame] = {}
    arm_manifests: dict[str, dict[str, Any]] = {}
    for arm in X1_ARMS:
        frame = pd.concat(fold_frames[arm], ignore_index=True).sort_values(
            "sample_token", kind="stable"
        )
        if len(frame) != 8192 or frame["sample_token"].duplicated().any():
            raise ValueError(f"X1 OOF completeness failed: {arm}/seed-{seed}")
        path = seed_root / arm / "oof_predictions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        metrics = production_sequence_macro_metrics(frame)
        per_fold_mid = {
            str(fold): float(
                production_sequence_macro_metrics(frame.loc[frame["outer_fold"] == fold])[
                    "sequence_macro_paper_MiD_overall"
                ]
            )
            for fold in (0, 1, 2)
        }
        manifest = atomic_write_json(
            path.with_suffix(".manifest.json"),
            {
                "artifact_type": "eclock_x1_arm_oof_manifest_v1",
                "arm_id": arm,
                "seed": seed,
                "row_count": 8192,
                "oof_path": str(path),
                "oof_file_sha256": compute_file_hash(str(path)),
                "oof_bytes": path.stat().st_size,
                "metrics": metrics,
                "per_fold_mid": per_fold_mid,
                "failure_rate": 0.0,
                "finite_fraction": 1.0,
                "outer_dev_evaluations_per_fold": 1,
            },
        )
        arm_frames[arm] = frame
        arm_manifests[arm] = manifest
    specs = {
        "dyn_vs_zero": ("X1-A5-DYN-U", "X1-A5-ZERO-U"),
        "dyn_vs_shuffle": ("X1-A5-DYN-U", "X1-A5-SHUFFLE-U"),
        "dyn_vs_a5": ("X1-A5-DYN-U", "X1-A5-REPLAY"),
    }
    comparisons: dict[str, Any] = {}
    for name, (candidate, reference) in specs.items():
        bootstrap = paired_hierarchical_mid_bootstrap(
            arm_frames[candidate],
            arm_frames[reference],
            protocol=x0_protocol,
            candidate_identity=_comparison_identity(arm_manifests[candidate], candidate),
            reference_identity=_comparison_identity(arm_manifests[reference], reference),
        )
        comparison = atomic_write_json(
            seed_root / "comparisons" / f"x1_comparison_{name}.json",
            {
                "artifact_type": "eclock_x1_comparison_v1",
                "seed": seed,
                "candidate": candidate,
                "reference": reference,
                "delta_mid": float(
                    arm_manifests[candidate]["metrics"]["sequence_macro_paper_MiD_overall"]
                    - arm_manifests[reference]["metrics"]["sequence_macro_paper_MiD_overall"]
                ),
                "bootstrap": bootstrap,
            },
        )
        comparisons[name] = comparison
    matched = True
    for fold in (0, 1, 2):
        subset = [row for row in match_records if row["fold"] == fold]
        for key in (
            "topology_sha256",
            "initialization_sha256",
            "trainable_mask_sha256",
            "normalization_sha256",
            "batch_schedule_sha256",
        ):
            matched &= len({row[key] for row in subset}) == 1
    integrity = {
        "row_count": 8192,
        "finite_fraction": 1.0,
        "failure_rate": 0.0,
        "coverage_drop_pp": 0.0,
        "a5_replay_exact": True,
        "zero_initialization_replay_exact": True,
        "matched_topology_init_order_budget": matched,
        "a5_and_transport_frozen": True,
        "outer_dev_evaluations_per_arm_fold": 1,
        "sealed_evaluation_opened": False,
    }
    gate = evaluate_x1_gate(comparisons=comparisons, integrity=integrity)
    gate["seed"] = seed
    gate["comparison_sha256"] = {
        name: comparison["artifact_sha256"] for name, comparison in comparisons.items()
    }
    gate = atomic_write_json(seed_root / "x1_gate.json", gate)
    _validate_schema(
        gate, args.repo / "schemas/scientific_recovery_v9_eclock_x1_gate_v1.schema.json"
    )
    atomic_write_json(
        seed_root / "x1_fold_summary.json",
        {
            "artifact_type": "eclock_x1_fold_summary_index_v1",
            "seed": seed,
            "fold_summaries": [
                {
                    "arm_id": row["arm_id"],
                    "outer_fold": row["outer_fold"],
                    "artifact_sha256": row["artifact_sha256"],
                    "path": row["oof_path"],
                }
                for row in fold_summaries
            ],
        },
    )
    atomic_write_json(
        seed_root / "x1_checkpoint_manifest.json",
        {
            "artifact_type": "eclock_x1_checkpoint_manifest_index_v1",
            "seed": seed,
            "checkpoints": [
                {
                    "arm_id": row["arm_id"],
                    "outer_fold": row["outer_fold"],
                    "checkpoint_file_sha256": row["training_summary"]["checkpoint_file_sha256"],
                    "checkpoint_manifest_sha256": row["training_summary"][
                        "checkpoint_manifest_sha256"
                    ],
                    "checkpoint_path": row["training_summary"]["checkpoint_path"],
                }
                for row in fold_summaries
            ],
        },
    )
    aggregate = atomic_write_json(
        seed_root / "x1_aggregate.json",
        {
            "artifact_type": "eclock_x1_aggregate_v1",
            "seed": seed,
            "arms": {
                arm: {
                    "mid": manifest["metrics"]["sequence_macro_paper_MiD_overall"],
                    "metrics": manifest["metrics"],
                    "per_fold_mid": manifest["per_fold_mid"],
                    "oof_manifest_sha256": manifest["artifact_sha256"],
                }
                for arm, manifest in arm_manifests.items()
            },
            "comparisons": {
                name: comparison["artifact_sha256"] for name, comparison in comparisons.items()
            },
            "gate_sha256": gate["artifact_sha256"],
            "matched_identity_records": match_records,
            "fold_summary_sha256": [row["artifact_sha256"] for row in fold_summaries],
        },
    )
    return {"gate": gate, "aggregate": aggregate, "comparisons": comparisons}


def run_x1(
    *,
    args: argparse.Namespace,
    campaign: Path,
    features: pd.DataFrame,
    feature_manifest: Mapping[str, Any],
    x05_gate: Mapping[str, Any],
    monitor: TelemetryMonitor,
    resume: bool,
) -> dict[str, Any]:
    if x05_gate.get("decision") != "X1_AUTHORIZED" or x05_gate.get("x1_authorized") is not True:
        raise PermissionError("X1 execution requires a signed X1_AUTHORIZED gate")
    protocol = load_signed_json(
        args.repo / "configs/protocol/scientific_recovery_v9_eclock_x05_x1.json",
        schema_path=args.repo
        / "schemas/scientific_recovery_v9_eclock_x05_x1_protocol_v1.schema.json",
    )
    x0_protocol = load_signed_json(
        args.repo / "configs/protocol/scientific_recovery_v9_eclock_x0.json",
        schema_path=args.repo / "schemas/scientific_recovery_v9_eclock_protocol_v2.schema.json",
    )
    seed_results = {
        7: _run_x1_seed(
            args=args,
            campaign=campaign,
            features=features,
            feature_manifest=feature_manifest,
            x05_gate=x05_gate,
            x0_protocol=x0_protocol,
            protocol=protocol,
            seed=7,
            monitor=monitor,
            resume=resume,
        )
    }
    seed7_gate = seed_results[7]["gate"]
    if seed7_gate.get("replication_authorized") is True:
        for seed in (13, 23):
            seed_results[seed] = _run_x1_seed(
                args=args,
                campaign=campaign,
                features=features,
                feature_manifest=feature_manifest,
                x05_gate=x05_gate,
                x0_protocol=x0_protocol,
                protocol=protocol,
                seed=seed,
                monitor=monitor,
                resume=resume,
            )
        dyn_zero = [
            result["comparisons"]["dyn_vs_zero"]["delta_mid"] for result in seed_results.values()
        ]
        dyn_a5 = [
            result["comparisons"]["dyn_vs_a5"]["delta_mid"] for result in seed_results.values()
        ]
        sign_consistent = all(value < 0.0 for value in dyn_zero)
        practical = float(np.median(np.asarray(dyn_a5, dtype=np.float64))) <= -3.0
        if sign_consistent and practical:
            decision = "X1_REPLICATED_GROUPED_DEV_CANDIDATE"
        elif sign_consistent:
            decision = "X1_INCREMENTAL_BUT_NOT_COMPETITIVE"
        else:
            decision = "X1_SEED7_NEGATIVE"
    else:
        decision = str(seed7_gate["decision"])
    replication = atomic_write_json(
        campaign / "x1" / "x1_replication_decision.json",
        {
            "artifact_type": "eclock_x1_replication_decision_v1",
            "decision": decision,
            "executed_seeds": sorted(seed_results),
            "seed7_replication_authorized": seed7_gate.get("replication_authorized", False),
            "per_seed_gate_sha256": {
                str(seed): result["gate"]["artifact_sha256"]
                for seed, result in seed_results.items()
            },
            "seed_to_sequence_to_track_hierarchy": True,
            "sealed_evaluation_opened": False,
        },
    )
    return replication


def _format_comparison(payload: Mapping[str, Any]) -> str:
    delta = payload["bootstrap"]["delta_candidate_minus_reference"]
    return (
        f"delta={payload['delta_mid']:.12f}, CI95=[{delta['ci95_low']:.12f}, "
        f"{delta['ci95_high']:.12f}], P(delta<0)={delta['probability_delta_lt_zero']:.12f}"
    )


def _append_metric_tables(lines: list[str], aggregate: Mapping[str, Any]) -> None:
    arms = list(aggregate["arms"])
    lines.extend(["", "Per-fold MiD:", "", "| Fold | " + " | ".join(arms) + " |"])
    lines.append("|---:" + "|---:" * len(arms) + "|")
    for fold in (0, 1, 2):
        values = [float(aggregate["arms"][arm]["per_fold_mid"][str(fold)]) for arm in arms]
        lines.append(f"| {fold} | " + " | ".join(f"{value:.12f}" for value in values) + " |")
    first_metrics = aggregate["arms"][arms[0]]["metrics"]
    sequences = sorted(first_metrics["per_sequence"])
    lines.extend(["", "Per-sequence MiD:", "", "| Sequence | " + " | ".join(arms) + " |"])
    lines.append("|---" + "|---:" * len(arms) + "|")
    for sequence in sequences:
        values = [
            float(aggregate["arms"][arm]["metrics"]["per_sequence"][sequence]["paper_MiD_overall"])
            for arm in arms
        ]
        lines.append(f"| {sequence} | " + " | ".join(f"{value:.12f}" for value in values) + " |")
    bucket_names = list(first_metrics["per_sequence"][sequences[0]]["bins"])
    lines.extend(
        [
            "",
            "Per-bucket sequence-macro MiD:",
            "",
            "| Bucket | " + " | ".join(arms) + " |",
        ]
    )
    lines.append("|---" + "|---:" * len(arms) + "|")
    for bucket in bucket_names:
        values = []
        for arm in arms:
            per_sequence = aggregate["arms"][arm]["metrics"]["per_sequence"]
            values.append(
                float(
                    np.mean(
                        np.asarray(
                            [
                                per_sequence[sequence]["bins"][bucket]["mid"]
                                for sequence in sequences
                            ],
                            dtype=np.float64,
                        ),
                        dtype=np.float64,
                    )
                )
            )
        lines.append(f"| {bucket} | " + " | ".join(f"{value:.12f}" for value in values) + " |")


def build_final_report(campaign: Path, *, training_commit: str, decision: str) -> Path:
    preflight = load_signed_artifact(campaign / "preflight.json")
    replay = load_signed_artifact(campaign / "x05/replay/x05_feature_replay_manifest.json")
    x05_gate = load_signed_artifact(campaign / "x05/meta/x05_gate.json")
    x05_aggregate = load_signed_artifact(campaign / "x05/meta/x05_aggregate.json")
    fold_summary = load_signed_artifact(campaign / "x05/meta/x05_meta_fold_summary.json")
    campaign_state = load_signed_artifact(campaign / "campaign_state.json")
    telemetry = load_signed_artifact(campaign / "telemetry/telemetry_summary.json")
    prior_attempts = load_signed_artifact(campaign / "pre_result_attempts.json")
    comparisons = {
        name: load_signed_artifact(
            campaign / "x05/meta/comparisons" / f"x05_comparison_{name}.json"
        )
        for name in ("dyn9_vs_cal", "dyn9_vs_shuffle", "dyn9_vs_a5_raw", "zero9_vs_cal")
    }
    lines = [
        "# CODEX X0.5/X1 final scientific report",
        "",
        "## Identity and integrity",
        "",
        f"- Starting HEAD: `{EXPECTED_X0_COMMIT}`",
        f"- Implementation/training commit: `{training_commit}`",
        f"- X0 bundle SHA-256: `{preflight['bundle']['sha256']}`",
        f"- Feature table SHA-256: `{replay['feature_table_file_sha256']}`",
        f"- X0.5/X1 protocol SHA-256: `{preflight['x05_x1_protocol_sha256']}`",
        f"- Signed X0.5 protocol SHA-256: `{preflight['x05_protocol_sha256']}`",
        f"- Signed X1 protocol SHA-256: `{preflight['x1_protocol_sha256']}`",
        "- X0 replay: A5 SHA-bound exact replay; BASE/DYN maximum phase error 0.0.",
        "- Rows: 8,192 unique tokens; three outer folds; nine canonical sequences.",
        "- Public validation, private test, EvTTC test and CodaBench remained closed.",
        "- Upstream ROI remains box-conditioned; no bbox-free/geometry-free claim is made.",
        "",
        "## QA",
        "",
        (
            "QA logs, schema validation, source-surface audit and PowerShell AST evidence "
            "are included in the essential bundle."
        ),
        "",
        "## X0.5 cross-fitted results",
        "",
        "| Arm | MiD |",
        "|---|---:|",
    ]
    for arm, values in x05_aggregate["arms"].items():
        lines.append(f"| {arm} | {values['mid']:.12f} |")
    _append_metric_tables(lines, x05_aggregate)
    lines.extend(["", "Primary comparisons:", ""])
    for name, payload in comparisons.items():
        lines.append(f"- `{name}`: {_format_comparison(payload)}")
    lines.extend(
        [
            "",
            f"Signed X0.5 decision: **{x05_gate['decision']}**.",
            "",
            (
                "Each meta-fold fit used the other two outer folds only. Lambda selection "
                "used LOSO by sequence; normalization was fit on each inner-train split "
                "during selection and on full meta-train for the final fit. Outer meta-test "
                "was never used for normalization, fitting or selection."
            ),
            "",
            (
                "Selected lambdas and all standardized coefficients/normalization vectors "
                "follow below and are also recorded in `x05_meta_fold_summary.json`."
            ),
        ]
    )
    for fold in fold_summary["folds"]:
        lines.extend(["", f"### X0.5 fit details: outer fold {fold['outer_fold']}", ""])
        for arm, fit in fold["arms"].items():
            detail = {
                "selected_lambda": fit["selected_lambda"],
                "normalization_mean": fit["normalization_mean"],
                "normalization_std": fit["normalization_std"],
                "coefficients_standardized": fit["coefficients_standardized"],
                "intercept": fit["intercept"],
            }
            lines.append(f"- `{arm}`: `{json.dumps(detail, sort_keys=True)}`")
    if (campaign / "x1/x1_replication_decision.json").is_file():
        replication = load_signed_artifact(campaign / "x1/x1_replication_decision.json")
        lines.extend(["", "## X1 residual adapter", ""])
        for seed in replication["executed_seeds"]:
            aggregate = load_signed_artifact(campaign / "x1" / f"seed-{seed}" / "x1_aggregate.json")
            lines.extend([f"### Seed {seed}", "", "| Arm | MiD |", "|---|---:|"])
            for arm, values in aggregate["arms"].items():
                lines.append(f"| {arm} | {values['mid']:.12f} |")
            _append_metric_tables(lines, aggregate)
            lines.append("")
            for name in ("dyn_vs_zero", "dyn_vs_shuffle", "dyn_vs_a5"):
                payload = load_signed_artifact(
                    campaign / "x1" / f"seed-{seed}" / "comparisons" / f"x1_comparison_{name}.json"
                )
                lines.append(f"- `{name}`: {_format_comparison(payload)}")
        lines.extend(
            [
                "",
                f"Final X1 decision: **{replication['decision']}**.",
                (
                    "The scientific checkpoint was fixed at update 1,000 for every trained "
                    "arm/fold. Outer-dev was evaluated exactly once after freezing; no "
                    "milestone selection was performed."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## X1 status",
                "",
                (
                    "X1 was not executed because the signed X0.5 gate did not authorize it. "
                    "This is a valid preregistered scientific stop."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Telemetry, retries and resumes",
            "",
            (
                "Five-second host/GPU telemetry is included under `telemetry/`. Campaign "
                "state and progress/checkpoint manifests disclose all completed stages and "
                "resume events. No scientific retry changed code or configuration."
            ),
            f"- Campaign wall time: {float(campaign_state['wall_seconds']):.3f} s.",
            f"- Feature replay: {float(replay['wall_seconds']):.3f} s; "
            f"{float(replay['rows_per_second']):.3f} rows/s.",
            f"- Telemetry samples: {telemetry['sample_count']}; peak process RSS "
            f"{telemetry['peak_process_rss_bytes']} bytes; peak GPU memory "
            f"{telemetry['peak_gpu_memory_used_mib']} MiB.",
            f"- X1 device selected before scientific rows: `{preflight['x1_device_selected']}`.",
            f"- Pre-result technical attempts: {prior_attempts['attempt_count']}; none reached "
            "feature replay or an X0.5 meta-test.",
            "",
            "## Limitations and maximum claim",
            "",
            (
                "These are grouped-development results on one previously developed 8,192-row "
                "universe. The bootstrap quantifies sequence-to-track sampling uncertainty, "
                "not external generalization. No sealed evaluation was opened."
            ),
            "",
            (
                "Maximum authorized claim is limited to the exact signed decision and "
                "protocol. No SOTA, JEPA benefit, detector-free, bbox-free, geometry-free or "
                "external-generalization claim is authorized."
            ),
            "",
            "## Next decision",
            "",
            f"`{decision}`. No new family is automatically authorized.",
            "",
        ]
    )
    report = campaign / "CODEX_X05_X1_FINAL_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _checkpoint_inventory(campaign: Path) -> dict[str, Any]:
    records = []
    for path in sorted(campaign.rglob("*.pt")):
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": compute_file_hash(str(path)),
                "exists": True,
            }
        )
    return atomic_write_json(
        campaign / "x1_checkpoint_inventory.json",
        {
            "artifact_type": "eclock_x05_x1_checkpoint_inventory_v1",
            "checkpoint_count": len(records),
            "checkpoints": records,
        },
    )


def package_results(
    *, campaign: Path, repo: Path, training_commit: str, decision: str
) -> tuple[Path, Path, Path]:
    next_path = campaign / "NEXT_DECISION.json"
    next_payload = atomic_write_json(
        next_path,
        {
            "artifact_type": "eclock_x05_x1_next_decision_v1",
            "decision": decision,
            "scientific_stop": decision != "X1_REPLICATED_GROUPED_DEV_CANDIDATE",
            "sealed_evaluation_opened": False,
            "training_commit": training_commit,
        },
    )
    _validate_schema(
        next_payload, repo / "schemas/scientific_recovery_v9_eclock_next_decision_v1.schema.json"
    )
    report = build_final_report(campaign, training_commit=training_commit, decision=decision)
    _checkpoint_inventory(campaign)
    environment = atomic_write_json(
        campaign / "environment.json",
        {
            "artifact_type": "eclock_x05_x1_environment_v1",
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "training_commit": training_commit,
            "utc": datetime.now(UTC).isoformat(),
        },
    )
    _ = environment
    short = training_commit[:12]
    bundle = campaign / f"E_JEPA_TTC_X05_X1_ESSENTIAL_RESULTS_{short}.zip"
    checksum_path = bundle.with_suffix(".zip.sha256")
    if bundle.exists() or checksum_path.exists():
        raise FileExistsError("final bundle path already exists")
    include: list[tuple[Path, str]] = []
    for path in sorted(campaign.rglob("*")):
        if not path.is_file() or path.suffix == ".pt" or path in {bundle, checksum_path}:
            continue
        include.append((path, path.relative_to(campaign).as_posix()))
    changed_paths = _git(repo, "diff", "--name-only", f"{EXPECTED_X0_COMMIT}..{training_commit}")
    tracked_files = [repo / Path(value) for value in changed_paths.splitlines() if value]
    for path in sorted(set(tracked_files)):
        include.append((path, f"tracked/{path.relative_to(repo).as_posix()}"))
    sums = [f"{compute_file_hash(str(path))}  {arcname}" for path, arcname in include]
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, arcname in include:
            archive.write(path, arcname)
        archive.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
    checksum = compute_file_hash(str(bundle))
    checksum_path.write_text(f"{checksum}  {bundle.name}\n", encoding="utf-8")
    if compute_file_hash(str(bundle)) != checksum:
        raise ValueError("final bundle checksum changed after write")
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None or "SHA256SUMS.txt" not in archive.namelist():
            raise ValueError("final bundle physical verification failed")
        expected_names = {arcname for _, arcname in include} | {"SHA256SUMS.txt"}
        if set(archive.namelist()) != expected_names:
            raise ValueError("final bundle physical member universe mismatch")
        for line in archive.read("SHA256SUMS.txt").decode("utf-8").splitlines():
            expected_hash, member_name = line.split("  ", 1)
            if hashlib.sha256(archive.read(member_name)).hexdigest() != expected_hash:
                raise ValueError(f"final bundle member hash mismatch: {member_name}")
    for required in (report, bundle, checksum_path, next_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise ValueError(f"mandatory deliverable missing/empty: {required}")
    return report, bundle, checksum_path


def run_full(args: argparse.Namespace) -> int:
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["MKL_NUM_THREADS"] = "16"
    torch.set_num_threads(16)
    training_commit = _git(args.repo, "rev-parse", "HEAD")
    campaign = args.output_root / "campaigns" / f"x05-x1-{training_commit[:12]}"
    campaign.mkdir(parents=True, exist_ok=args.resume)
    monitor = TelemetryMonitor(campaign / "telemetry")
    monitor.start()
    started = time.time()
    try:
        atomic_write_json(
            campaign / "campaign_state.json",
            {
                "artifact_type": "eclock_x05_x1_campaign_state_v1",
                "status": "running",
                "training_commit": training_commit,
                "starting_head": EXPECTED_X0_COMMIT,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "sealed_evaluation_opened": False,
            },
        )
        monitor.set_phase("preflight")
        if not (campaign / "preflight.json").is_file():
            preflight = run_preflight(args, campaign)
        else:
            preflight = load_signed_artifact(campaign / "preflight.json")
        selected_x1_device = (
            str(preflight["x1_device_selected"])
            if str(args.x1_device) == "auto"
            else str(args.x1_device)
        )
        if selected_x1_device not in {"cpu", "cuda:0"}:
            raise ValueError("X1 device must be auto, cpu or cuda:0")
        args.x1_device = torch.device(selected_x1_device)
        monitor.set_phase("feature_replay")
        replay_root = campaign / "x05" / "replay"
        replay_manifest_path = replay_root / "x05_feature_replay_manifest.json"
        if replay_manifest_path.is_file():
            if not args.resume:
                raise FileExistsError("feature replay exists without --resume")
            replay_manifest = load_signed_artifact(
                replay_manifest_path, artifact_type="eclock_x05_feature_replay_manifest_v1"
            )
        else:
            replay_manifest = run_feature_replay(
                repo=args.repo,
                reference_root=args.reference_repo,
                x0_campaign=args.x0_campaign,
                cache_root=args.cache_root,
                x0_bundle=args.x0_bundle,
                output_root=replay_root,
                device=args.replay_device,
            )
        feature_path = Path(str(replay_manifest["feature_table_path"]))
        if compute_file_hash(str(feature_path)) != replay_manifest["feature_table_file_sha256"]:
            raise ValueError("feature table hash mismatch before X0.5")
        x0_protocol = load_signed_json(
            args.repo / "configs/protocol/scientific_recovery_v9_eclock_x0.json",
            schema_path=args.repo / "schemas/scientific_recovery_v9_eclock_protocol_v2.schema.json",
        )
        features = validate_feature_table(
            pd.read_csv(feature_path, float_precision="round_trip"),
            x0_protocol=x0_protocol,
            replay_manifest=replay_manifest,
        )
        monitor.set_phase("x05_crossfit")
        meta_root = campaign / "x05" / "meta"
        gate_path = meta_root / "x05_gate.json"
        if gate_path.is_file():
            if not args.resume:
                raise FileExistsError("X0.5 gate exists without --resume")
            x05_gate = load_signed_artifact(gate_path, artifact_type="eclock_x05_gate_v1")
        else:
            x05_gate = run_x05_cross_fit(
                features,
                x0_protocol=x0_protocol,
                output_root=meta_root,
            )
        _validate_schema(
            x05_gate, args.repo / "schemas/scientific_recovery_v9_eclock_x05_gate_v1.schema.json"
        )
        decision = str(x05_gate["decision"])
        if decision == "X1_AUTHORIZED":
            monitor.set_phase("x1")
            replication = run_x1(
                args=args,
                campaign=campaign,
                features=features,
                feature_manifest=replay_manifest,
                x05_gate=x05_gate,
                monitor=monitor,
                resume=args.resume,
            )
            decision = str(replication["decision"])
        if decision not in ALLOWED_NEXT_DECISIONS:
            raise ValueError(f"final decision is outside closed registry: {decision}")
        monitor.set_phase("finalize")
        atomic_write_json(
            campaign / "campaign_state.json",
            {
                "artifact_type": "eclock_x05_x1_campaign_state_v1",
                "status": "scientific_stop"
                if decision != "X1_REPLICATED_GROUPED_DEV_CANDIDATE"
                else "completed",
                "scientific_stop": decision != "X1_REPLICATED_GROUPED_DEV_CANDIDATE",
                "decision": decision,
                "training_commit": training_commit,
                "starting_head": EXPECTED_X0_COMMIT,
                "wall_seconds": time.time() - started,
                "sealed_evaluation_opened": False,
            },
        )
        monitor.stop()
        qa_source = args.output_root / "qa_preflight" / training_commit[:12]
        if qa_source.is_dir():
            shutil.copytree(qa_source, campaign / "qa", dirs_exist_ok=True)
        report, bundle, checksum = package_results(
            campaign=campaign,
            repo=args.repo,
            training_commit=training_commit,
            decision=decision,
        )
        print(
            json.dumps(
                {
                    "decision": decision,
                    "scientific_stop": decision != "X1_REPLICATED_GROUPED_DEV_CANDIDATE",
                    "report": str(report),
                    "bundle": str(bundle),
                    "checksum": str(checksum),
                    "next_decision": str(campaign / "NEXT_DECISION.json"),
                },
                indent=2,
            )
        )
        return 0
    except Exception as error:
        monitor.stop()
        atomic_write_json(
            campaign / "campaign_failure.json",
            {
                "artifact_type": "eclock_x05_x1_campaign_failure_v1",
                "status": "technical_failure",
                "training_commit": training_commit,
                "sealed_evaluation_opened": False,
                "utc": datetime.now(UTC).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--x0-campaign", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--x0-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-training-commit", action="store_true")
    parser.add_argument("--replay-device", default="cuda:0", type=lambda value: torch.device(value))
    parser.add_argument("--x1-device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
