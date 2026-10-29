"""Label-blind, sealed inference for the ten-sequence EvTTC competition download."""

# ruff: noqa: E402 -- bind this checkout before importing project modules
# PyTorch exposes several runtime attributes dynamically.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from e_jepa_ttc.data import eap_representation, event_v4_geometry
from e_jepa_ttc.efficient_context import mapped_union
from e_jepa_ttc.representations import voxel_grid
from e_jepa_ttc.rgb_port import data as rgb_data
from e_jepa_ttc.simplex_t import phase
from operational.evttc_competition.geometry import Calibration, load_calibration
from operational.evttc_competition.prepare import POLICY, prepare_query
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_competition.runtime import CompetitionRuntime, RuntimePrediction
from operational.rgb_port_streaming import contracts
from scripts.final_train40_transfer import verify_candidate, verify_frozen_runtime

SEQUENCES = (
    "CCRs1-low",
    "CCRs1-medium",
    "CCRs1-high",
    "CCRs2-low",
    "CCRs2-medium",
    "CCRs2-high",
    "CCRm-low",
    "CCRm-medium",
    "Slider-750",
    "Slider-1000",
)
EXPECTED_ROWS = 434
RESULT_COLUMNS = ("Index", "timestamp (s)", "ttc (s)", "cost time(s)")
RUN_SCHEMA = "evttc_competition_inference_freeze_v2"
ROW_SCHEMA = "evttc_competition_row_v1"
PARENT_ATTEMPT_REASON = (
    "V2 replaces the failed first attempt with revised causal preparation/runtime support; "
    "the parent remains immutable and is used only as provenance."
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "as_tuple") and value.__class__.__name__ == "Decimal":
        return str(value)
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def freeze_json(path: Path, value: Any, *, resume: bool) -> str:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"Frozen run already exists; pass --resume: {path}")
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Resume freeze differs byte-for-byte: {path}")
    else:
        atomic_text(path, encoded)
    return sha256_file(path)


def calibration_record(calibration: Calibration) -> dict[str, Any]:
    return {
        "event_intrinsics": calibration.event_intrinsics.tolist(),
        "event_distortion": calibration.event_distortion.tolist(),
        "event_resolution": list(calibration.event_resolution),
        "rgb_intrinsics": calibration.rgb_intrinsics.tolist(),
        "rgb_distortion": calibration.rgb_distortion.tolist(),
        "rgb_resolution": list(calibration.rgb_resolution),
        "rotation_event_to_rgb": calibration.rotation_event_to_rgb.tolist(),
        "source_kind": calibration.source_kind,
        "transform_source": calibration.transform_source,
        "translation_omitted": calibration.translation_omitted,
        "metadata": json_safe(calibration.metadata),
    }


def parent_attempt_record(path: Path) -> dict[str, Any]:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Parent attempt is not a directory: {root}")
    freeze = root / "RUN_FREEZE.json"
    errors = root / "errors.jsonl"
    rows = root / "all_rows.jsonl"
    if not freeze.is_file() or not errors.is_file() or not rows.is_file():
        raise ValueError(
            "Parent attempt must contain RUN_FREEZE.json, all_rows.jsonl, and errors.jsonl"
        )
    return {
        "path_resolved": str(root),
        "run_freeze_sha256": sha256_file(freeze),
        "all_rows_ledger_sha256": sha256_file(rows),
        "errors_ledger_sha256": sha256_file(errors),
        "reason": PARENT_ATTEMPT_REASON,
        "read_only": True,
    }


def query_identity(sequence_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "row_index": int(row["row_index"]),
        "index": int(row["index"]),
        "timestamp_text": str(row["timestamp_text"]),
        "anchor_us": int(row["anchor_us"]),
        "box_text": [str(item) for item in row["box_text"]],
    }


def identity_key(identity: Mapping[str, Any]) -> str:
    return f"{identity['sequence_id']}:{int(identity['row_index']):06d}"


def preflight_inputs(
    data_root: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Calibration],
]:
    rows_by_sequence: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    calibrations: dict[str, Calibration] = {}
    for sequence_id in SEQUENCES:
        with SensorSequence(data_root, sequence_id) as reader:
            rows = reader.bbox_rows()
            if not rows:
                raise ValueError(f"No competition bbox queries found for {sequence_id}")
            rows_by_sequence[sequence_id] = rows
            sources[sequence_id] = json_safe(reader.source_metadata())
        kind = "slider_candidate" if sequence_id.startswith("Slider-") else "road_right"
        calibrations[sequence_id] = load_calibration(data_root / sequence_id, kind=kind)
    count = sum(len(rows) for rows in rows_by_sequence.values())
    if count != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} competition rows, found {count}")
    return rows_by_sequence, sources, calibrations


def source_hashes() -> dict[str, str]:
    modules = (
        sys.modules[__name__],
        sys.modules[SensorSequence.__module__],
        sys.modules[load_calibration.__module__],
        sys.modules[prepare_query.__module__],
        sys.modules[CompetitionRuntime.__module__],
        sys.modules[verify_candidate.__module__],
        event_v4_geometry,
        mapped_union,
        eap_representation,
        voxel_grid,
        rgb_data,
        contracts,
        phase,
    )
    paths = {Path(inspect.getfile(module)).resolve() for module in modules}
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in sorted(paths)
    }


def build_freeze(
    *,
    data_root: Path,
    candidate: Path,
    device: str,
    prepare_only: bool,
    rows_by_sequence: Mapping[str, Sequence[Mapping[str, Any]]],
    source_metadata: Mapping[str, Mapping[str, Any]],
    calibrations: Mapping[str, Calibration],
    parent_attempt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _paths, model_binding = verify_candidate(candidate)
    frozen_runtime = verify_frozen_runtime(candidate)
    query_manifest = {
        sequence_id: [query_identity(sequence_id, row) for row in rows_by_sequence[sequence_id]]
        for sequence_id in SEQUENCES
    }
    calibration_manifest = {
        sequence_id: calibration_record(calibrations[sequence_id]) for sequence_id in SEQUENCES
    }
    return {
        "schema": RUN_SCHEMA,
        "model": model_binding,
        "frozen_runtime": frozen_runtime,
        "device": device,
        "run_mode": "prepare_only" if prepare_only else "inference",
        "seed": 7,
        "precision": "float32",
        "policy": POLICY,
        "policy_sha256": canonical_sha256(POLICY),
        "query_manifest": query_manifest,
        "query_manifest_sha256": canonical_sha256(query_manifest),
        "calibrations": calibration_manifest,
        "calibrations_sha256": canonical_sha256(calibration_manifest),
        "raw_sources": source_metadata,
        "raw_sources_sha256": canonical_sha256(source_metadata),
        "executed_source_sha256": source_hashes(),
        "data_root_resolved": str(data_root),
        "candidate_root_resolved": str(candidate),
        "parent_attempt": parent_attempt,
        "sequences": list(SEQUENCES),
        "queries": EXPECTED_ROWS,
        "targets_read": False,
        "training_updates": 0,
        "checkpoint_selection": False,
        "runtime_boundary": {
            "included_per_query": [
                "sensor_query_preprocessing",
                "model_gpu_or_cpu_core",
                "host_scalar_selection_and_postprocessing",
            ],
            "excluded_one_time": [
                "model_load",
                "seal_preflight",
                "raw_index_scan",
                "source_hashing",
                "calibration_freeze",
            ],
        },
        "limitations": [
            "The downloaded bbox CSV is treated as the author query contract, but this "
            "mapping is not independently verified.",
            "The official parser acceptance of signed TTC values has not been verified; "
            "signed finite predictions are preserved without clipping.",
            "Road and Slider cross-camera geometry use declared rotation-only mappings "
            "with translation omitted because object depth is unavailable.",
        ],
    }


def load_existing_rows(
    path: Path, freeze_sha256: str, *, resume: bool
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    if not resume:
        raise FileExistsError(f"Existing row ledger requires --resume: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != ROW_SCHEMA or row.get("freeze_sha256") != freeze_sha256:
                raise ValueError(f"Row ledger freeze mismatch at line {line_number}")
            if row.get("status") not in {"ok", "unavailable", "prepared_only"}:
                raise RuntimeError(
                    f"Non-canonical resumed row at line {line_number}; inspect the old "
                    "failure and start a new output"
                )
            key = identity_key(row["identity"])
            if key in records:
                raise ValueError(f"Duplicate resumed row identity: {key}")
            records[key] = row
    return records


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def runtime_fields(result: RuntimePrediction, wall_seconds: float) -> dict[str, Any]:
    return {
        "model_wall_seconds": wall_seconds,
        "runtime_internal_timing_scope": (
            "host_spans_with_synchronous_output_extraction_not_CUDA_event_timings"
        ),
        "runtime_internal": json_safe(asdict(result)),
    }


def run_rows(
    *,
    data_root: Path,
    candidate: Path,
    output: Path,
    device: str,
    rows_by_sequence: Mapping[str, list[dict[str, Any]]],
    calibrations: Mapping[str, Calibration],
    freeze_sha256: str,
    prepare_only: bool,
    limit: int | None,
    resume: bool,
) -> list[dict[str, Any]]:
    ledger = output / "all_rows.jsonl"
    error_ledger = output / "errors.jsonl"
    if resume and error_ledger.exists() and error_ledger.stat().st_size:
        raise RuntimeError(
            "Cannot resume a run with errors.jsonl; inspect the failure and start a new "
            "output after correcting the frozen inputs or code"
        )
    records = load_existing_rows(ledger, freeze_sha256, resume=resume)
    runtime = None if prepare_only else CompetitionRuntime.from_candidate(candidate, device=device)
    fresh = 0
    for sequence_id in SEQUENCES:
        rows = rows_by_sequence[sequence_id]
        with SensorSequence(data_root, sequence_id) as reader:
            for row in rows:
                identity = query_identity(sequence_id, row)
                key = identity_key(identity)
                if key in records:
                    continue
                if limit is not None and fresh >= limit:
                    continue
                started = time.perf_counter()
                try:
                    prepared = prepare_query(reader, rows, row, calibrations[sequence_id])
                    preprocess_seconds = time.perf_counter() - started
                    base = {
                        "schema": ROW_SCHEMA,
                        "freeze_sha256": freeze_sha256,
                        "identity": identity,
                        "input_details": json_safe(prepared.details),
                        "preprocess_seconds": preprocess_seconds,
                    }
                    if prepare_only:
                        record = {
                            **base,
                            "status": "prepared_only",
                            "prediction_ttc_seconds": None,
                            "cost_time_seconds": preprocess_seconds,
                            "reason": "prepare_only_requested",
                        }
                    else:
                        assert runtime is not None
                        synchronize(device)
                        model_started = time.perf_counter()
                        prediction = runtime.predict(
                            prepared.events, prepared.rgb, int(row["anchor_us"])
                        )
                        synchronize(device)
                        model_wall = time.perf_counter() - model_started
                        timings = runtime_fields(prediction, model_wall)
                        cost = preprocess_seconds + model_wall
                        record = {
                            **base,
                            **timings,
                            "status": prediction.status,
                            "prediction_ttc_seconds": prediction.model_ttc,
                            "cost_time_seconds": cost,
                            "reason": prediction.model_reason,
                            "prediction": json_safe(asdict(prediction)),
                        }
                        if prediction.status == "ok" and (
                            prediction.model_ttc is None or not math.isfinite(prediction.model_ttc)
                        ):
                            raise ValueError("Runtime returned an ok status without a finite TTC")
                except Exception as error:
                    elapsed = time.perf_counter() - started
                    failure = {
                        "schema": ROW_SCHEMA,
                        "freeze_sha256": freeze_sha256,
                        "identity": identity,
                        "status": "error",
                        "prediction_ttc_seconds": None,
                        "cost_time_seconds": elapsed,
                        "reason": "unexpected_exception_fail_fast",
                        "error": {"type": type(error).__name__, "message": str(error)},
                    }
                    append_row(error_ledger, failure)
                    raise
                append_row(ledger, record)
                records[key] = record
                fresh += 1
                print(
                    json.dumps(
                        {"completed_fresh": fresh, "query": key, "status": record["status"]}
                    ),
                    flush=True,
                )
    return list(records.values())


def expected_queries(
    rows_by_sequence: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[tuple[int, str]]]:
    return {
        sequence_id: [
            (int(row["index"]), str(row["timestamp_text"])) for row in rows_by_sequence[sequence_id]
        ]
        for sequence_id in SEQUENCES
    }


def verify_frozen_inputs_unchanged(
    *,
    data_root: Path,
    candidate: Path,
    frozen_model: Mapping[str, Any],
    frozen_runtime: Mapping[str, Any],
    frozen_parent_attempt: Mapping[str, Any] | None,
    frozen_sources: Mapping[str, Mapping[str, Any]],
    frozen_code_hashes: Mapping[str, str],
) -> None:
    """Close the preflight-to-package TOCTOU window for code and raw inputs."""
    current_code = source_hashes()
    if current_code != dict(frozen_code_hashes):
        raise RuntimeError("Executed source bytes changed after the run freeze")
    _paths, current_model = verify_candidate(candidate)
    if current_model != dict(frozen_model):
        raise RuntimeError("Sealed candidate bytes or binding changed after the run freeze")
    current_runtime = verify_frozen_runtime(candidate)
    if current_runtime != dict(frozen_runtime):
        raise RuntimeError("Frozen runtime archive or source parity changed after the run freeze")
    if frozen_parent_attempt is not None:
        parent_root = Path(str(frozen_parent_attempt["path_resolved"]))
        if parent_attempt_record(parent_root) != dict(frozen_parent_attempt):
            raise RuntimeError("Parent attempt provenance changed after the run freeze")
    current_sources: dict[str, dict[str, Any]] = {}
    for sequence_id in SEQUENCES:
        with SensorSequence(data_root, sequence_id) as reader:
            current_sources[sequence_id] = json_safe(reader.source_metadata())
    if current_sources != dict(frozen_sources):
        raise RuntimeError("Raw sensor, bbox, or calibration bytes changed after the run freeze")


def validate_signed_results(
    root: Path,
    expected: Mapping[str, Sequence[tuple[int, str]]],
) -> dict[str, Any]:
    errors: list[str] = []
    files: list[dict[str, Any]] = []
    actual_sequences = {
        path.parent.name for path in root.glob("*/results.txt") if path.parent.parent == root
    }
    if actual_sequences != set(expected):
        errors.append(
            "Results sequence set mismatch: "
            f"expected {sorted(expected)}, found {sorted(actual_sequences)}"
        )
    for sequence_id, identities in expected.items():
        path = root / sequence_id / "results.txt"
        if not path.is_file():
            errors.append(f"Missing results file: {sequence_id}")
            continue
        with path.open("r", encoding="utf-8", newline="") as stream:
            content = list(csv.reader(stream, delimiter="\t"))
        if not content or tuple(content[0]) != RESULT_COLUMNS:
            errors.append(f"Invalid columns: {sequence_id}")
            continue
        parsed: list[tuple[int, str, float, float]] = []
        try:
            for values in content[1:]:
                if len(values) != 4:
                    raise ValueError("row does not have four columns")
                parsed.append((int(values[0]), values[1], float(values[2]), float(values[3])))
        except ValueError as error:
            errors.append(f"Invalid row in {sequence_id}: {error}")
            continue
        identity = [(row[0], row[1]) for row in parsed]
        if identity != list(identities):
            errors.append(f"Official query identity mismatch: {sequence_id}")
        if any(not math.isfinite(row[2]) or not math.isfinite(row[3]) for row in parsed):
            errors.append(f"Non-finite value: {sequence_id}")
        if any(row[3] < 0 for row in parsed):
            errors.append(f"Negative runtime: {sequence_id}")
        files.append(
            {
                "sequence_id": sequence_id,
                "path": path.relative_to(root).as_posix(),
                "rows": len(parsed),
                "sha256": sha256_file(path),
            }
        )
    return {"valid": not errors, "errors": errors, "files": files}


def write_signed_results(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Sequence[tuple[int, str]]],
) -> dict[str, Any]:
    by_key = {identity_key(row["identity"]): row for row in records}
    for sequence_id, identities in expected.items():
        path = root / sequence_id / "results.txt"
        lines = ["\t".join(RESULT_COLUMNS)]
        for row_index, (index, timestamp_text) in enumerate(identities):
            key = f"{sequence_id}:{row_index:06d}"
            record = by_key.get(key)
            if record is None or record.get("status") != "ok":
                raise ValueError(f"Cannot write incomplete result identity: {key}")
            prediction = float(record["prediction_ttc_seconds"])
            cost = float(record["cost_time_seconds"])
            if not math.isfinite(prediction) or not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid signed result values: {key}")
            if (
                record["identity"]["index"] != index
                or record["identity"]["timestamp_text"] != timestamp_text
            ):
                raise ValueError(f"Record identity differs from frozen query: {key}")
            lines.append(f"{index}\t{timestamp_text}\t{prediction:.17g}\t{cost:.17g}")
        atomic_text(path, "\n".join(lines) + "\n")
    report = validate_signed_results(root, expected)
    if not report["valid"]:
        raise ValueError(f"Signed result validation failed: {report['errors']}")
    return report


def deterministic_zip(path: Path, results_root: Path, files: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for entry in sorted(files, key=lambda item: str(item["path"])):
            relative = str(entry["path"])
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (results_root / relative).read_bytes())
    os.replace(temporary, path)


def write_partial_results(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Sequence[tuple[int, str]]],
) -> dict[str, Any]:
    """Export finite rows under names that cannot be mistaken for an official submission."""
    by_key = {identity_key(row["identity"]): row for row in records}
    files: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    available_count = 0
    for sequence_id, identities in expected.items():
        path = root / sequence_id / "available_predictions.tsv"
        lines = ["\t".join(RESULT_COLUMNS)]
        sequence_available = 0
        for row_index, (index, timestamp_text) in enumerate(identities):
            key = f"{sequence_id}:{row_index:06d}"
            record = by_key.get(key)
            if record is None:
                raise ValueError(f"Terminal partial ledger is missing identity: {key}")
            identity = record["identity"]
            if identity["index"] != index or identity["timestamp_text"] != timestamp_text:
                raise ValueError(f"Terminal partial identity differs from freeze: {key}")
            value = record.get("prediction_ttc_seconds")
            cost = float(record["cost_time_seconds"])
            if record.get("status") == "ok" and value is not None:
                prediction = float(value)
                if not math.isfinite(prediction) or not math.isfinite(cost) or cost < 0:
                    raise ValueError(f"Invalid available partial values: {key}")
                lines.append(f"{index}\t{timestamp_text}\t{prediction:.17g}\t{cost:.17g}")
                available_count += 1
                sequence_available += 1
            else:
                unavailable.append(
                    {
                        "sequence_id": sequence_id,
                        "row_index": row_index,
                        "index": index,
                        "timestamp_text": timestamp_text,
                        "status": record.get("status"),
                        "reason": record.get("reason"),
                    }
                )
        atomic_text(path, "\n".join(lines) + "\n")
        files.append(
            {
                "sequence_id": sequence_id,
                "path": path.relative_to(root).as_posix(),
                "available_rows": sequence_available,
                "sha256": sha256_file(path),
            }
        )
    notice = root / "PARTIAL_NOT_SUBMISSION.txt"
    atomic_text(
        notice,
        "PARTIAL RESULTS ONLY\n"
        "This directory is not an official-complete EvTTC submission.\n"
        "Only finite available predictions are listed; unavailable rows are in the manifest.\n",
    )
    files.append(
        {
            "path": notice.relative_to(root).as_posix(),
            "sha256": sha256_file(notice),
        }
    )
    manifest = {
        "schema": "evttc_competition_partial_results_v1",
        "status": "PARTIAL_NOT_SUBMISSION",
        "expected_rows": sum(len(rows) for rows in expected.values()),
        "available_rows": available_count,
        "unavailable_rows": unavailable,
        "sequence_files": files[:-1],
    }
    manifest_path = root / "partial_manifest.json"
    atomic_json(manifest_path, manifest)
    files.append(
        {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
        }
    )
    return {**manifest, "files": files}


def git_identity() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def finalize(
    *,
    output: Path,
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Sequence[tuple[int, str]]],
    freeze_sha256: str,
    device: str,
    prepare_only: bool,
    limit: int | None,
    started_utc: str,
) -> dict[str, Any]:
    expected_count = sum(len(rows) for rows in expected.values())
    ok = [row for row in records if row.get("status") == "ok"]
    nulls = [row for row in records if row.get("prediction_ttc_seconds") is None]
    zero_support_rows = sum(
        any(
            bool(observation.get("zero_support"))
            for observation in row.get("input_details", {}).get("event_observations", [])
        )
        for row in records
    )
    known_diagnostics: dict[str, dict[str, int]] = {}
    for row in records:
        internal = row.get("runtime_internal", {})
        for modality, values in internal.get("producer_diagnostics", {}).items():
            aggregate = known_diagnostics.setdefault(
                str(modality), {"known_true": 0, "known_total": 0, "support_count": 0}
            )
            for key in tuple(aggregate):
                aggregate[key] += int(values.get(key, 0))
    complete = (
        not prepare_only
        and limit is None
        and len(records) == expected_count
        and len(ok) == expected_count
        and not nulls
    )
    terminal_partial = (
        not prepare_only and limit is None and len(records) == expected_count and not complete
    )
    summary: dict[str, Any] = {
        "status": "COMPLETE" if complete else "PARTIAL",
        "expected_rows": expected_count,
        "recorded_rows": len(records),
        "ok_rows": len(ok),
        "null_rows": len(nulls),
        "terminal_population_accounted": len(records) == expected_count,
        "available_prediction_rows": len(ok),
        "zero_support_rows": zero_support_rows,
        "producer_known_mask_diagnostics": known_diagnostics,
        "freeze_sha256": freeze_sha256,
        "prepare_only": prepare_only,
        "limit": limit,
        "targets_read": False,
        "submission_or_email_performed": False,
    }
    if not complete:
        archive = (output / "results.zip").resolve()
        results_root = (output / "results").resolve()
        if archive.parent != output.resolve() or results_root.parent != output.resolve():
            raise RuntimeError("Refusing to clean results outside the run output")
        if archive.exists():
            archive.unlink()
        if results_root.exists():
            shutil.rmtree(results_root)
        partial_root = (output / "partial_results").resolve()
        partial_archive = (output / "partial_results.zip").resolve()
        if partial_root.parent != output.resolve() or partial_archive.parent != output.resolve():
            raise RuntimeError("Refusing to manage partial results outside the run output")
        if partial_root.exists():
            shutil.rmtree(partial_root)
        if partial_archive.exists():
            partial_archive.unlink()
        if terminal_partial:
            partial = write_partial_results(partial_root, records, expected)
            deterministic_zip(partial_archive, partial_root, partial["files"])
            summary.update(
                partial_results="partial_results",
                partial_results_zip="partial_results.zip",
                partial_results_zip_sha256=sha256_file(partial_archive),
                partial_available_rows=partial["available_rows"],
                partial_unavailable_rows=len(partial["unavailable_rows"]),
            )
        marker = (
            "PARTIAL\n"
            f"expected_rows={expected_count}\nrecorded_rows={len(records)}\n"
            f"ok_rows={len(ok)}\nnull_rows={len(nulls)}\n"
            "No complete submission ZIP was written.\n"
        )
        atomic_text(output / "results_available.txt", marker)
    else:
        marker = output / "results_available.txt"
        if marker.exists():
            marker.unlink()
        results_root = output / "results"
        partial_root = output / "partial_results"
        partial_archive = output / "partial_results.zip"
        if partial_root.exists():
            shutil.rmtree(partial_root)
        if partial_archive.exists():
            partial_archive.unlink()
        validation = write_signed_results(results_root, records, expected)
        archive = output / "results.zip"
        deterministic_zip(archive, results_root, validation["files"])
        summary.update(
            results_validation=validation,
            results_zip="results.zip",
            results_zip_sha256=sha256_file(archive),
        )
    git_commit, git_dirty = git_identity()
    environment = {
        "start_utc": started_utc,
        "end_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "dirty_source_sha256": canonical_sha256(source_hashes()),
        "seed": 7,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "device": device,
    }
    manifest = {
        "schema": "evttc_competition_submission_manifest_v1",
        "status": summary["status"],
        "freeze_sha256": freeze_sha256,
        "results_zip_sha256": summary.get("results_zip_sha256"),
        "partial_results_zip_sha256": summary.get("partial_results_zip_sha256"),
        "parser_contract": {
            "columns": list(RESULT_COLUMNS),
            "delimiter": "tab",
            "signed_finite_ttc_preserved": True,
            "official_signed_value_acceptance_verified": False,
        },
        "author_query_contract_verified": False,
        "limitations": [
            "Signed TTC parser acceptance is unverified; no clipping or absolute value "
            "was applied.",
            "The bbox CSV row order and timestamp text are treated as the author query "
            "contract without independent confirmation.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "environment.json", environment)
    atomic_json(output / "summary.json", summary)
    report = [
        "# EvTTC competition inference",
        "",
        f"Status: **{summary['status']}**",
        f"Rows: {len(records)} recorded / {expected_count} expected; {len(ok)} finite predictions.",
        "",
        "The run was label-blind and preserved signed TTC predictions without clipping.",
        "Official acceptance of signed values and the author query contract remain unverified.",
        (
            "A distinctly named partial-results archive was emitted and is not a submission."
            if terminal_partial
            else "No partial-results archive was emitted."
        ),
        "One-time model loading, preflight, index scanning, source hashing, and "
        "calibration freezing are excluded from per-query cost.",
    ]
    atomic_text(output / "report.md", "\n".join(report) + "\n")
    return summary


def main() -> None:
    started_utc = datetime.now(UTC).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parent-attempt", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    data_root = args.data_root.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    parent_attempt = (
        parent_attempt_record(args.parent_attempt) if args.parent_attempt is not None else None
    )

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)

    rows_by_sequence, metadata, calibrations = preflight_inputs(data_root)
    freeze = build_freeze(
        data_root=data_root,
        candidate=candidate,
        device=args.device,
        prepare_only=args.prepare_only,
        rows_by_sequence=rows_by_sequence,
        source_metadata=metadata,
        calibrations=calibrations,
        parent_attempt=parent_attempt,
    )
    freeze_sha = freeze_json(output / "RUN_FREEZE.json", freeze, resume=args.resume)
    records = run_rows(
        data_root=data_root,
        candidate=candidate,
        output=output,
        device=args.device,
        rows_by_sequence=rows_by_sequence,
        calibrations=calibrations,
        freeze_sha256=freeze_sha,
        prepare_only=args.prepare_only,
        limit=args.limit,
        resume=args.resume,
    )
    verify_frozen_inputs_unchanged(
        data_root=data_root,
        candidate=candidate,
        frozen_model=freeze["model"],
        frozen_runtime=freeze["frozen_runtime"],
        frozen_parent_attempt=freeze["parent_attempt"],
        frozen_sources=freeze["raw_sources"],
        frozen_code_hashes=freeze["executed_source_sha256"],
    )
    summary = finalize(
        output=output,
        records=records,
        expected=expected_queries(rows_by_sequence),
        freeze_sha256=freeze_sha,
        device=args.device,
        prepare_only=args.prepare_only,
        limit=args.limit,
        started_utc=started_utc,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
