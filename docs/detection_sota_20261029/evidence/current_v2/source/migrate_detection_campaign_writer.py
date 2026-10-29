"""Fail-closed R0 checkpoint migration for the streaming-JSON writer revision.

This tool never edits an existing run.  It accepts only the frozen fcad2299 R0v2
producer and a newly frozen runtime whose sole trainer AST change is
``write_json``.  Execution still requires a separate PASS memory-profile receipt.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import gc
import hashlib
import io
import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

OLD_GIT_COMMIT = "fcad2299979766f6d175e45b8016ea3b7a11a594"
OLD_TRAINER_SHA256 = "168b1041de16e1d6adce10f00cdce6b2e7c02ecbdee2ef08e0a4f6a9c00bc8da"
NEW_TRAINER_SHA256 = "8c24aa9ed57db34d499f6e81d93637b0fa2de401e3ff844964df3df3bfe073bf"
OLD_WRITER_SHA256 = "d6193c5bfb099d29f56d24ee5616e1bd6a1f196a472d5a5a438b10d4bd5f5729"
NEW_WRITER_SHA256 = "425c080dddc4ee2ab326af8f939d56ce15b0603981bca96c139b83b8711d6f25"
TRAINER_RELATIVE = "scripts/train_detection_campaign.py"
CHECKPOINT_SCHEMA = "eap_detection_campaign_checkpoint_v1"
AUTHORIZED_PROFILE_SHA256: str | None = (
    "a892e08d6048332a0f2928a43131ee7d7b3f3375d8d81e52f8562fff1f7f31d4"
)
AUTHORIZED_FAILED_SOURCE_RECEIPT_SHA256: str | None = None


def sha256_file(path: Path) -> str:
    """Hash one file without materializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value canonically."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verified_evidence(profile_path: Path, profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Hash and load every independently reviewed profiler receipt."""
    required = {
        "serialization_1024",
        "stream_worker_4096",
        "stream_watchdog_4096",
        "legacy_worker_4096",
        "legacy_watchdog_4096",
        "admission_receipt",
        "metric_parity_receipt",
    }
    raw = profile.get("evidence")
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("Streaming-writer profile evidence population differs")
    result = {}
    root = profile_path.parent.resolve()
    for name, entry in raw.items():
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"Malformed profile evidence entry: {name}")
        relative = entry["path"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"Unsafe profile evidence path: {name}")
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"Profile evidence bytes differ: {name}")
        result[name] = _load_json(path)
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _writer_node(source: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "write_json"
    ]
    if len(matches) != 1:
        raise ValueError("Trainer must contain exactly one top-level write_json")
    return matches[0]


def _writer_hash(source: str) -> str:
    node = _writer_node(source)
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError("Could not isolate write_json source")
    return hashlib.sha256(segment.encode()).hexdigest()


def _normalized_module_ast(source: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_json"
    ]
    if len(matches) != 1:
        raise ValueError("Trainer must contain exactly one top-level write_json")
    matches[0].body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def verify_writer_only_ast(old_path: Path, new_path: Path) -> dict[str, str]:
    """Prove the frozen trainer differs only inside the allowlisted writer body."""
    old_source = old_path.read_text(encoding="utf-8")
    new_source = new_path.read_text(encoding="utf-8")
    observed = {
        "old_trainer_sha256": sha256_file(old_path),
        "new_trainer_sha256": sha256_file(new_path),
        "old_writer_sha256": _writer_hash(old_source),
        "new_writer_sha256": _writer_hash(new_source),
    }
    expected = {
        "old_trainer_sha256": OLD_TRAINER_SHA256,
        "new_trainer_sha256": NEW_TRAINER_SHA256,
        "old_writer_sha256": OLD_WRITER_SHA256,
        "new_writer_sha256": NEW_WRITER_SHA256,
    }
    if observed != expected:
        raise ValueError(f"Trainer/writer hashes leave the allowlist: {observed}")
    if _normalized_module_ast(old_source) != _normalized_module_ast(new_source):
        raise ValueError("Trainer AST differs outside write_json")
    return observed | {
        "normalized_module_ast_sha256": canonical_sha256(_normalized_module_ast(old_source))
    }


def serializer_bytes(value: object, *, streaming: bool) -> bytes:
    """Serialize with the old or new implementation for parity fixtures."""
    if not streaming:
        return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    stream = io.StringIO()
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write("\n")
    return stream.getvalue().encode()


def verify_serializer_parity(value: object) -> dict[str, object]:
    """Require byte-for-byte equivalence, not merely equivalent parsed JSON."""
    old, new = serializer_bytes(value, streaming=False), serializer_bytes(value, streaming=True)
    if old != new:
        raise ValueError("Streaming writer bytes differ from the frozen writer")
    return {"byte_count": len(old), "sha256": hashlib.sha256(old).hexdigest(), "status": "PASS"}


def _verify_profile(path: Path) -> dict[str, Any]:
    if AUTHORIZED_PROFILE_SHA256 is None or sha256_file(path) != AUTHORIZED_PROFILE_SHA256:
        raise ValueError(
            "Streaming-writer profile is not the pinned independently reviewed receipt"
        )
    profile = _load_json(path)
    required = {
        "schema": "eap_detection_streaming_writer_profile_v1",
        "status": "PASS",
        "old_writer_sha256": OLD_WRITER_SHA256,
        "new_writer_sha256": NEW_WRITER_SHA256,
        "serializer_byte_parity": True,
        "metric_parity": True,
        "peak_memory_admission": True,
        "profile_scope": "observed_1024_4096_projected_23958",
        "observed_parity_sample_counts": [1024, 4096],
        "streaming_rss_observed_sample_count": 4096,
        "full_population_sample_count": 23958,
        "full_population_observed": False,
        "projection_is_measurement": False,
        "projection_multiplier": 1.5,
        "memory_basis": "physical_ram_not_commit_or_pagefile",
        "evaluator_measurement_excludes_trainer_model": True,
        "admission_mode": "migration_before_model_load_total",
        "independent_review_status": "PASS",
    }
    if any(profile.get(key) != value for key, value in required.items()):
        raise ValueError("Streaming-writer profile has not passed the fixed admission gate")
    evidence = _verified_evidence(path, profile)
    for name in (
        "serialization_1024",
        "stream_worker_4096",
        "stream_watchdog_4096",
        "legacy_worker_4096",
        "legacy_watchdog_4096",
        "metric_parity_receipt",
    ):
        if evidence[name].get("status") != "PASS":
            raise ValueError(f"Profile evidence is not PASS: {name}")
    if evidence["admission_receipt"].get("status") not in {
        "CONDITIONALLY_ADMIT_STREAMING_ONLY",
        "PASS",
    }:
        raise ValueError("Streaming admission evidence is not admitted")
    for field in ("serializer_sha256_by_sample_count", "metrics_sha256_by_sample_count"):
        panels = profile.get(field)
        if not isinstance(panels, Mapping) or set(panels) != {"1024", "4096"}:
            raise ValueError(f"Streaming-writer profile lacks observed parity panels: {field}")
        for sample_count, pair in panels.items():
            if not isinstance(pair, Mapping) or set(pair) != {"old", "new"}:
                raise ValueError(f"Malformed parity pair for {field}/{sample_count}")
            old_digest, new_digest = pair["old"], pair["new"]
            if not isinstance(old_digest, str) or len(old_digest) != 64 or old_digest != new_digest:
                raise ValueError(f"Streaming-writer parity digest failed: {field}/{sample_count}")
    comparison = evidence["serialization_1024"]
    stream_worker = evidence["stream_worker_4096"]
    legacy_worker = evidence["legacy_worker_4096"]
    stream_watchdog = evidence["stream_watchdog_4096"]
    legacy_watchdog = evidence["legacy_watchdog_4096"]
    parity = evidence["metric_parity_receipt"]
    admission = evidence["admission_receipt"]
    if (
        comparison.get("frame_count") != 1024
        or comparison.get("byte_exact_match") is not True
        or comparison.get("payload_sha256_both_modes")
        != profile["serializer_sha256_by_sample_count"]["1024"]["old"]
        or any(worker.get("frame_count") != 4096 for worker in (stream_worker, legacy_worker))
        or stream_worker.get("serialized_sha256")
        != profile["serializer_sha256_by_sample_count"]["4096"]["new"]
        or legacy_worker.get("serialized_sha256")
        != profile["serializer_sha256_by_sample_count"]["4096"]["old"]
        or parity.get("serializer_sha256_by_sample_count")
        != profile["serializer_sha256_by_sample_count"]
        or parity.get("metrics_sha256_by_sample_count") != profile["metrics_sha256_by_sample_count"]
    ):
        raise ValueError("Profile payload/metric evidence differs from parity claims")
    for field in ("profiler_sha256", "source_selection_sha256", "protocol_sha256"):
        value = profile.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Profile lacks source identity: {field}")
    if (
        any(
            worker.get("executed_script_sha256") != profile["profiler_sha256"]
            or worker.get("source_sha256") != profile["source_selection_sha256"]
            or worker.get("protocol_sha256") != profile["protocol_sha256"]
            for worker in (stream_worker, legacy_worker)
        )
        or admission.get("executed_profiler_sha256") != profile["profiler_sha256"]
        or admission.get("source_selection_sha256") != profile["source_selection_sha256"]
    ):
        raise ValueError("Profiler/source/protocol identities differ across evidence")
    for watchdog_name, watchdog, worker_name in (
        ("stream_watchdog_4096", stream_watchdog, "stream_worker_4096"),
        ("legacy_watchdog_4096", legacy_watchdog, "legacy_worker_4096"),
    ):
        if (
            watchdog.get("returncode") != 0
            or watchdog.get("abort_reason") is not None
            or not watchdog.get("observed_owned_pids")
            or watchdog.get("worker_receipt_sha256") != profile["evidence"][worker_name]["sha256"]
        ):
            raise ValueError(f"Watchdog ownership/PASS evidence differs: {watchdog_name}")
    memory_fields = (
        "measured_evaluator_peak_rss_bytes",
        "fixed_trainer_model_cache_rss_bytes",
        "projected_full_evaluator_peak_rss_bytes",
        "projected_full_total_physical_ram_bytes",
        "observed_prestart_physical_free_ram_bytes",
        "observed_runtime_min_physical_free_ram_bytes",
        "physical_safety_margin_bytes",
    )
    if any(
        isinstance(profile.get(name), bool)
        or not isinstance(profile.get(name), int)
        or profile[name] <= 0
        for name in memory_fields
    ):
        raise ValueError("Streaming-writer profile physical-memory fields are malformed")
    observed_evaluator = profile["measured_evaluator_peak_rss_bytes"]
    fixed = profile["fixed_trainer_model_cache_rss_bytes"]
    projected_evaluator = math.ceil(observed_evaluator * (23958 / 4096) * 1.5)
    projected_total = fixed + projected_evaluator
    if (
        profile["projected_full_evaluator_peak_rss_bytes"] != projected_evaluator
        or profile["projected_full_total_physical_ram_bytes"] != projected_total
        or stream_worker.get("process_peak_rss_bytes") != observed_evaluator
        or legacy_watchdog.get("prestart_host_available_bytes")
        != profile["observed_prestart_physical_free_ram_bytes"]
        or legacy_watchdog.get("minimum_host_available_bytes")
        != profile["observed_runtime_min_physical_free_ram_bytes"]
        or admission.get("projection_23958", {})
        .get("fixed_trainer_model_cache", {})
        .get("conservative_allowance_bytes")
        != fixed
        or not isinstance(
            admission.get("projection_23958", {})
            .get("fixed_trainer_model_cache", {})
            .get("windows_peak_working_set_bytes"),
            int,
        )
        or admission["projection_23958"]["fixed_trainer_model_cache"][
            "windows_peak_working_set_bytes"
        ]
        > fixed
    ):
        raise ValueError("Streaming-writer full-population projection arithmetic is inconsistent")
    projection_fields = {
        key: profile[key]
        for key in (
            "streaming_rss_observed_sample_count",
            "full_population_sample_count",
            "projection_multiplier",
            "measured_evaluator_peak_rss_bytes",
            "evaluator_measurement_excludes_trainer_model",
            "fixed_trainer_model_cache_rss_bytes",
            "projected_full_evaluator_peak_rss_bytes",
            "projected_full_total_physical_ram_bytes",
            "observed_prestart_physical_free_ram_bytes",
            "observed_runtime_min_physical_free_ram_bytes",
            "physical_safety_margin_bytes",
            "memory_basis",
            "projection_is_measurement",
            "admission_mode",
        )
    }
    reviewer = profile.get("independent_reviewer")
    if (
        not isinstance(reviewer, str)
        or not reviewer
        or profile.get("reviewed_projection_sha256") != canonical_sha256(projection_fields)
    ):
        raise ValueError("Streaming-writer memory projection lacks independent review binding")
    safety = profile["physical_safety_margin_bytes"]
    required_prestart = projected_total + safety
    required_runtime = projected_evaluator + safety
    if (
        required_prestart > profile["observed_prestart_physical_free_ram_bytes"]
        or required_runtime > profile["observed_runtime_min_physical_free_ram_bytes"]
    ):
        raise ValueError("Streaming-writer profile memory arithmetic failed admission")
    current_physical = available_physical_ram_bytes()
    if required_prestart > current_physical:
        raise ValueError("Current physical RAM is below the reviewed migration admission")
    profile["migration_time_physical_free_ram_bytes"] = current_physical
    return profile


def available_physical_ram_bytes() -> int:
    """Read physical RAM headroom only; pagefile/commit is not an admission substitute."""
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.available_physical)
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def _verify_preregistration(path: Path, code_root: Path) -> dict[str, Any]:
    prereg = _load_json(path)
    code_hashes = prereg.get("code_sha256")
    portable_hashes = prereg.get("portable_source_sha256")
    if not isinstance(code_hashes, Mapping) or not isinstance(portable_hashes, Mapping):
        raise ValueError("Runtime preregistration lacks source hash manifests")
    if prereg.get("code_sha256_manifest") != canonical_sha256(code_hashes):
        raise ValueError("Runtime code manifest hash is inconsistent")
    if prereg.get("portable_source_sha256_manifest") != canonical_sha256(portable_hashes):
        raise ValueError("Portable source manifest hash is inconsistent")
    for relative, expected in code_hashes.items():
        source = code_root / str(relative)
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"Frozen runtime code differs: {relative}")
    portable_root = code_root / "portable_source"
    observed_portable = {
        source.relative_to(portable_root).as_posix(): sha256_file(source)
        for source in sorted(portable_root.rglob("*"))
        if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc"
    }
    if observed_portable != dict(portable_hashes):
        raise ValueError("Frozen portable source differs from preregistration")
    return prereg


def _runtime_provenance(prereg_path: Path, prereg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "runtime_preregistration_sha256": sha256_file(prereg_path),
        "source_receipt_sha256": prereg["source_receipt_sha256"],
        "effective_model_sha256": prereg["effective_model_sha256"],
        "code_sha256_manifest": prereg["code_sha256_manifest"],
        "portable_source_sha256_manifest": prereg["portable_source_sha256_manifest"],
        "matmul_precision": prereg["matmul_precision"],
    }


def verify_runtime_transition(
    old_prereg_path: Path,
    old_code_root: Path,
    new_prereg_path: Path,
    new_code_root: Path,
    new_trainer_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Allow only the trainer writer hash and its aggregate code hash to change."""
    old = _verify_preregistration(old_prereg_path, old_code_root)
    new = _verify_preregistration(new_prereg_path, new_code_root)
    ast_receipt = verify_writer_only_ast(old_code_root / TRAINER_RELATIVE, new_trainer_path)
    if sha256_file(new_code_root / TRAINER_RELATIVE) != NEW_TRAINER_SHA256:
        raise ValueError("New frozen code does not contain the allowlisted trainer")
    invariant_fields = {
        "schema",
        "config",
        "config_sha256",
        "protocol_sha256",
        "source_receipt",
        "source_receipt_sha256",
        "effective_model",
        "effective_model_sha256",
        "portable_source_sha256",
        "portable_source_sha256_manifest",
        "dataset_media_sha256",
        "matmul_precision",
    }
    for field in invariant_fields:
        if old.get(field) != new.get(field):
            raise ValueError(f"Runtime invariant changed during writer migration: {field}")
    old_code, new_code = dict(old["code_sha256"]), dict(new["code_sha256"])
    if set(old_code) != set(new_code) or old_code.get(TRAINER_RELATIVE) != OLD_TRAINER_SHA256:
        raise ValueError("Old/new runtime code populations differ")
    changed = {key for key in old_code if old_code[key] != new_code[key]}
    if changed != {TRAINER_RELATIVE} or new_code[TRAINER_RELATIVE] != NEW_TRAINER_SHA256:
        raise ValueError(f"Runtime files changed outside the writer allowlist: {sorted(changed)}")
    return old, new, ast_receipt


def _tensor_descriptor(value: torch.Tensor) -> dict[str, object]:
    contiguous = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(memoryview(cast(Any, contiguous))).hexdigest()
    return {
        "kind": "tensor",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "requires_grad": value.requires_grad,
        "logical_bytes_sha256": digest,
    }


def _fingerprint(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_descriptor(value)
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": [
                [_fingerprint(key), _fingerprint(nested)]
                for key, nested in sorted(value.items(), key=lambda item: repr(item[0]))
            ],
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_fingerprint(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_fingerprint(item) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": type(value).__name__, "value": value}
    raise TypeError(f"Unsupported checkpoint state type: {type(value).__name__}")


def _state_without_runtime(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"runtime_provenance", "runtime_provenance_sha256"}
    }


def numerical_state_sha256(payload: Mapping[str, Any]) -> str:
    """Hash every tensor and scalar outside the two provenance fields."""
    return canonical_sha256(_fingerprint(_state_without_runtime(payload)))


def _assert_exact(left: Any, right: Any, path: str = "checkpoint") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ValueError(f"State type differs at {path}")
        if _tensor_descriptor(left) != _tensor_descriptor(right) or not torch.equal(left, right):
            raise ValueError(f"Tensor state differs at {path}")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if (
            not isinstance(left, Mapping)
            or not isinstance(right, Mapping)
            or set(left) != set(right)
        ):
            raise ValueError(f"Mapping state differs at {path}")
        for key in left:
            _assert_exact(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"Sequence state differs at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_exact(left_item, right_item, f"{path}[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise ValueError(f"Scalar state differs at {path}")


def migrate_checkpoint(
    source: Path,
    destination: Path,
    *,
    old_provenance: Mapping[str, Any],
    new_provenance: Mapping[str, Any],
) -> dict[str, object]:
    """Copy one checkpoint while changing only its bound runtime provenance."""
    old = torch.load(source, map_location="cpu", weights_only=True)
    if (
        old.get("schema") != CHECKPOINT_SCHEMA
        or old.get("producer") != TRAINER_RELATIVE
        or old.get("runtime_provenance") != dict(old_provenance)
        or old.get("runtime_provenance_sha256") != canonical_sha256(old_provenance)
    ):
        raise ValueError(f"Checkpoint is not bound to the admitted old runtime: {source}")
    source_sha256 = sha256_file(source)
    before_fingerprint = _fingerprint(_state_without_runtime(old))
    before = canonical_sha256(before_fingerprint)
    old["runtime_provenance"] = dict(new_provenance)
    old["runtime_provenance_sha256"] = canonical_sha256(new_provenance)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(old, destination)
    del old
    gc.collect()
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    if restored.get("runtime_provenance") != dict(new_provenance):
        raise ValueError("Migrated checkpoint did not retain new provenance")
    after_fingerprint = _fingerprint(_state_without_runtime(restored))
    after = canonical_sha256(after_fingerprint)
    if before_fingerprint != after_fingerprint or before != after:
        raise ValueError("Migrated numerical checkpoint state changed")
    return {
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "destination": destination.name,
        "destination_sha256": sha256_file(destination),
        "update": restored["update"],
        "best_metric": restored["best_metric"],
        "numerical_state_sha256": before,
        "status": "PASS_EXACT_STATE_PROVENANCE_ONLY",
    }


def _failed_source_lineage(old_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the failed run and disclose completed-but-nondurable updates."""
    failure_path = old_run / "failure.json"
    preregistration_path = old_run / "preregistration.json"
    metrics_path = old_run / "metrics.jsonl"
    if not all(path.is_file() for path in (failure_path, preregistration_path, metrics_path)):
        raise ValueError("Writer migration requires FAILED receipt, preregistration and metrics")
    failure = _load_json(failure_path)
    if failure.get("status") != "FAILED":
        raise ValueError("Writer migration source must retain its FAILED receipt")
    preregistration = _load_json(preregistration_path)
    parent_chain = preregistration.get("parent_run_chain")
    if not isinstance(parent_chain, list):
        raise ValueError("Failed source preregistration lacks its parent run chain")
    updates: list[int] = []
    with metrics_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            update = record.get("update") if isinstance(record, Mapping) else None
            if isinstance(update, bool) or not isinstance(update, int) or update < 1:
                raise ValueError(f"Malformed update in metrics.jsonl line {line_number}")
            updates.append(update)
    if not updates or any(
        right != left + 1 for left, right in zip(updates, updates[1:], strict=False)
    ):
        raise ValueError("Failed source metrics updates are empty or non-contiguous")
    return failure, {
        "failure_receipt": {
            "path": str(failure_path.resolve()),
            "sha256": sha256_file(failure_path),
            "error_type": failure.get("error_type"),
            "error": failure.get("error"),
        },
        "preregistration_sha256": sha256_file(preregistration_path),
        "parent_run": preregistration.get("parent_run"),
        "parent_run_chain": parent_chain,
        "metrics_jsonl_sha256": sha256_file(metrics_path),
        "first_logged_update": updates[0],
        "last_logged_update": updates[-1],
        "logged_update_count": len(updates),
    }


def _verify_source_checkpoint_authority(
    path: Path,
    *,
    old_run: Path,
    last_source: Path,
    selected_source: Path,
    durable_update: int,
    best_metric: object,
    source_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a separately reviewed byte authority for a failed source run."""
    if (
        AUTHORIZED_FAILED_SOURCE_RECEIPT_SHA256 is None
        or sha256_file(path) != AUTHORIZED_FAILED_SOURCE_RECEIPT_SHA256
    ):
        raise ValueError("Failed source checkpoint authority is not pinned in this tool")
    authority = _load_json(path)
    selected_expected = (
        {
            "status": "NOT_AVAILABLE_NO_SCHEDULED_SELECTION",
            "sha256": None,
            "best_metric": None,
        }
        if best_metric is None
        else {
            "status": "AVAILABLE",
            "sha256": sha256_file(selected_source),
            "best_metric": best_metric,
        }
    )
    expected = {
        "schema": "eap_detection_failed_checkpoint_authority_v1",
        "status": "PASS_INDEPENDENT_REVIEW",
        "old_run": str(old_run.resolve()),
        "failure_receipt_sha256": source_lineage["failure_receipt"]["sha256"],
        "preregistration_sha256": source_lineage["preregistration_sha256"],
        "runtime_preregistration_sha256": sha256_file(old_run / "runtime_preregistration.json"),
        "metrics_jsonl_sha256": source_lineage["metrics_jsonl_sha256"],
        "durable_last_checkpoint_sha256": sha256_file(last_source),
        "durable_update": durable_update,
        "last_logged_update": source_lineage["last_logged_update"],
        "selected_checkpoint": selected_expected,
    }
    if any(authority.get(key) != value for key, value in expected.items()):
        raise ValueError("Failed source checkpoint authority differs from source bytes")
    reviewer = authority.get("independent_reviewer")
    if not isinstance(reviewer, str) or not reviewer:
        raise ValueError("Failed source checkpoint authority lacks reviewer identity")
    return authority


def migrate(
    *,
    old_run: Path,
    producer_receipt_path: Path,
    new_prereg_path: Path,
    new_code_root: Path,
    new_trainer_path: Path,
    stream_profile_path: Path,
    source_checkpoint_receipt_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Create a new migration run; all inputs remain immutable."""
    if output.exists():
        raise FileExistsError(f"Migration output already exists: {output}")
    profile = _verify_profile(stream_profile_path)
    _, source_lineage = _failed_source_lineage(old_run)
    producer = _load_json(producer_receipt_path)
    if (
        producer.get("schema") != "detection_campaign_producer_freeze_v1"
        or producer.get("git_commit") != OLD_GIT_COMMIT
        or producer.get("source_sha256", {}).get(TRAINER_RELATIVE) != OLD_TRAINER_SHA256
    ):
        raise ValueError("Producer receipt is not the sole admitted fcad2299 R0v2 source")
    old_prereg_path = old_run / "runtime_preregistration.json"
    old_code_root = old_run / "code"
    old_prereg, new_prereg, ast_receipt = verify_runtime_transition(
        old_prereg_path,
        old_code_root,
        new_prereg_path,
        new_code_root,
        new_trainer_path,
    )
    if producer.get("source_sha256") != old_prereg.get("code_sha256"):
        raise ValueError("Producer receipt source population differs from the old runtime")
    old_provenance = _runtime_provenance(old_prereg_path, old_prereg)
    new_provenance = _runtime_provenance(new_prereg_path, new_prereg)
    last_source = old_run / "checkpoints/last.pt"
    selected_source = old_run / "checkpoints/selected.pt"
    if not last_source.is_file():
        raise ValueError("Migration requires a durable last.pt")
    last_metadata = torch.load(last_source, map_location="cpu", weights_only=True)
    durable_update = last_metadata.get("update")
    best_metric = last_metadata.get("best_metric")
    del last_metadata
    gc.collect()
    if isinstance(durable_update, bool) or not isinstance(durable_update, int):
        raise ValueError("Durable checkpoint update is malformed")
    if durable_update > source_lineage["last_logged_update"]:
        raise ValueError("Durable checkpoint exceeds the failed metrics journal")
    if best_metric is None:
        if selected_source.exists():
            raise ValueError("best_metric=None cannot have a selected checkpoint")
        selected_status: dict[str, object] = {
            "status": "NOT_AVAILABLE_NO_SCHEDULED_SELECTION",
            "source": None,
            "destination": None,
            "best_metric": None,
        }
    else:
        if not selected_source.is_file():
            raise ValueError("A non-null best metric requires selected.pt")
        selected_status = {}
    source_lineage["durable_resume_update"] = durable_update
    source_lineage["completed_but_not_durable"] = {
        "recoverable": False,
        "first_update": durable_update + 1,
        "last_update": source_lineage["last_logged_update"],
        "count": source_lineage["last_logged_update"] - durable_update,
        "disposition": "discarded; resume exclusively from durable last.pt",
    }
    source_authority = _verify_source_checkpoint_authority(
        source_checkpoint_receipt_path,
        old_run=old_run,
        last_source=last_source,
        selected_source=selected_source,
        durable_update=durable_update,
        best_metric=best_metric,
        source_lineage=source_lineage,
    )

    temporary = output.with_name(f".{output.name}.migrating.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Migration temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(new_code_root, temporary / "code")
        shutil.copy2(new_prereg_path, temporary / "runtime_preregistration.json")
        ancestor_dir = temporary / "ancestor_receipts"
        ancestor_dir.mkdir()
        ancestor_files: dict[str, dict[str, object]] = {}
        for source in sorted(path for path in old_run.iterdir() if path.is_file()):
            destination = ancestor_dir / source.name
            shutil.copy2(source, destination)
            ancestor_files[source.name] = {
                "source": str(source.resolve()),
                "source_sha256": sha256_file(source),
                "copied_sha256": sha256_file(destination),
            }
        checkpoints = {
            "last": migrate_checkpoint(
                last_source,
                temporary / "checkpoints/last.pt",
                old_provenance=old_provenance,
                new_provenance=new_provenance,
            )
        }
        if selected_source.is_file():
            checkpoints["selected"] = migrate_checkpoint(
                selected_source,
                temporary / "checkpoints/selected.pt",
                old_provenance=old_provenance,
                new_provenance=new_provenance,
            )
        else:
            checkpoints["selected"] = selected_status
        receipt = {
            "schema": "eap_detection_campaign_writer_migration_v1",
            "status": "PREPARED_NOT_EXECUTED_SCIENTIFICALLY",
            "old_git_commit": OLD_GIT_COMMIT,
            "old_run": str(old_run.resolve()),
            "producer_receipt": {
                "path": str(producer_receipt_path.resolve()),
                "sha256": sha256_file(producer_receipt_path),
            },
            "profile": {
                "path": str(stream_profile_path.resolve()),
                "sha256": sha256_file(stream_profile_path),
                "receipt": profile,
            },
            "ast_proof": ast_receipt,
            "old_runtime_preregistration_sha256": sha256_file(old_prereg_path),
            "new_runtime_preregistration_sha256": sha256_file(new_prereg_path),
            "old_runtime_provenance_sha256": canonical_sha256(old_provenance),
            "new_runtime_provenance_sha256": canonical_sha256(new_provenance),
            "runtime_invariants_sha256": canonical_sha256(
                {
                    key: old_prereg[key]
                    for key in (
                        "config_sha256",
                        "protocol_sha256",
                        "source_receipt_sha256",
                        "effective_model_sha256",
                        "portable_source_sha256_manifest",
                        "dataset_media_sha256",
                        "matmul_precision",
                    )
                }
            ),
            "ancestor_receipts": ancestor_files,
            "failed_source_lineage": source_lineage,
            "source_checkpoint_authority": {
                "path": str(source_checkpoint_receipt_path.resolve()),
                "sha256": sha256_file(source_checkpoint_receipt_path),
                "receipt": source_authority,
            },
            "checkpoints": checkpoints,
        }
        _write_json(temporary / "migration_receipt.json", receipt)
        os.replace(temporary, output)
        return receipt
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--new-runtime-preregistration", type=Path, required=True)
    parser.add_argument("--new-code-root", type=Path, required=True)
    parser.add_argument("--new-trainer", type=Path, required=True)
    parser.add_argument("--stream-profile", type=Path, required=True)
    parser.add_argument("--source-checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = migrate(
        old_run=args.old_run,
        producer_receipt_path=args.producer_receipt,
        new_prereg_path=args.new_runtime_preregistration,
        new_code_root=args.new_code_root,
        new_trainer_path=args.new_trainer,
        stream_profile_path=args.stream_profile,
        source_checkpoint_receipt_path=args.source_checkpoint_receipt,
        output=args.output,
    )
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
