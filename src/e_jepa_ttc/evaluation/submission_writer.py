"""Strict EvTTC results writer and offline validator."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from e_jepa_ttc.utils.io import write_structured

SUBMISSION_COLUMNS = ("Index", "timestamp (s)", "ttc (s)", "cost time(s)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_sequence_id(sequence_id: str) -> str:
    if not sequence_id or sequence_id in {".", ".."}:
        raise ValueError("sequence_id cannot be empty or relative.")
    if any(character in sequence_id for character in ("/", "\\", ":")):
        raise ValueError("sequence_id must be a single safe path component.")
    return sequence_id


def write_sequence_results(
    output_root: str | Path,
    *,
    sequence_id: str,
    indices: list[int],
    timestamps: list[int | float],
    ttc_seconds: list[float],
    cost_time_seconds: list[float],
) -> dict[str, Any]:
    """Write one tab-separated official-results candidate without changing predictions."""

    sequence = _safe_sequence_id(sequence_id)
    count = len(indices)
    if not all(len(values) == count for values in (timestamps, ttc_seconds, cost_time_seconds)):
        raise ValueError("All submission columns must contain the same number of rows.")
    if count == 0:
        raise ValueError("A sequence submission cannot be empty.")
    if len(set(indices)) != count or indices != sorted(indices):
        raise ValueError("Indices must be unique and sorted.")
    if any(
        not math.isfinite(float(value))
        for values in (timestamps, ttc_seconds, cost_time_seconds)
        for value in values
    ):
        raise ValueError("Submission values must be finite.")
    if any(value <= 0 for value in ttc_seconds):
        raise ValueError("Every predicted TTC must be positive.")
    if any(value < 0 for value in cost_time_seconds):
        raise ValueError("Every runtime must be non-negative.")
    root = Path(output_root)
    path = root / sequence / "results.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(SUBMISSION_COLUMNS)
        for row in zip(indices, timestamps, ttc_seconds, cost_time_seconds, strict=True):
            writer.writerow(
                (
                    str(int(row[0])),
                    format(float(row[1]), ".17g"),
                    format(float(row[2]), ".17g"),
                    format(float(row[3]), ".17g"),
                )
            )
    return {
        "sequence_id": sequence,
        "path": path.relative_to(root).as_posix(),
        "rows": count,
        "sha256": _sha256(path),
    }


def write_submission_manifest(
    output_root: str | Path,
    *,
    candidate_name: str,
    sequence_files: list[dict[str, Any]],
    checkpoint_sha256: str,
    freeze_manifest_sha256: str,
    runtime_environment: dict[str, Any],
) -> dict[str, Any]:
    """Sign the immutable mapping between predictions, candidate and runtime."""

    if not sequence_files:
        raise ValueError("sequence_files cannot be empty.")
    root = Path(output_root)
    payload: dict[str, Any] = {
        "format": "evttc_tabulated_results_v1",
        "columns": list(SUBMISSION_COLUMNS),
        "delimiter": "tab",
        "candidate_name": candidate_name,
        "checkpoint_sha256": checkpoint_sha256,
        "freeze_manifest_sha256": freeze_manifest_sha256,
        "sequence_files": sorted(sequence_files, key=lambda row: row["sequence_id"]),
        "runtime_includes": [
            "online_window_and_voxelization",
            "encoder",
            "queries",
            "geometry",
            "postprocessing",
            "all_ensemble_members_when_applicable",
        ],
        "runtime_excludes": ["one_time_model_load", "checkpoint_read"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_structured(root / "submission_manifest.json", payload)
    write_structured(root / "runtime_environment.json", runtime_environment)
    return payload


def validate_submission_root(
    submission_root: str | Path,
    *,
    expected_queries: dict[str, list[tuple[int, float]]] | None = None,
    require_sequences: int | None = None,
) -> dict[str, Any]:
    """Validate format, values, hashes and optional official query identity."""

    root = Path(submission_root)
    manifest_path = root / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    sequence_files = manifest.get("sequence_files", [])
    if require_sequences is not None and len(sequence_files) != require_sequences:
        errors.append(f"Expected {require_sequences} sequences, found {len(sequence_files)}.")
    for declared in sequence_files:
        sequence_id = str(declared["sequence_id"])
        path = root / str(declared["path"])
        if not path.is_file():
            errors.append(f"Missing results file: {path}")
            continue
        if _sha256(path) != declared["sha256"]:
            errors.append(f"Hash mismatch: {sequence_id}")
        with path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.reader(source, delimiter="\t")
            content = list(reader)
        if not content or tuple(content[0]) != SUBMISSION_COLUMNS:
            errors.append(f"Invalid columns: {sequence_id}")
            continue
        parsed: list[tuple[int, float, float, float]] = []
        try:
            for values in content[1:]:
                if len(values) != 4:
                    raise ValueError("row does not have four columns")
                parsed.append(
                    (int(values[0]), float(values[1]), float(values[2]), float(values[3]))
                )
        except ValueError as error:
            errors.append(f"Invalid row in {sequence_id}: {error}")
            continue
        indices = [row[0] for row in parsed]
        timestamps = [row[1] for row in parsed]
        if len(set(indices)) != len(indices) or indices != sorted(indices):
            errors.append(f"Duplicate or unsorted indices: {sequence_id}")
        if len(set(timestamps)) != len(timestamps) or timestamps != sorted(timestamps):
            errors.append(f"Duplicate or unsorted timestamps: {sequence_id}")
        if any(not all(math.isfinite(value) for value in row[1:]) for row in parsed):
            errors.append(f"Non-finite value: {sequence_id}")
        if any(row[2] <= 0 or row[3] < 0 for row in parsed):
            errors.append(f"Non-positive TTC or negative runtime: {sequence_id}")
        if expected_queries is not None:
            expected = expected_queries.get(sequence_id)
            identity = [(row[0], row[1]) for row in parsed]
            if expected is None or identity != expected:
                errors.append(f"Official query identity mismatch: {sequence_id}")
        rows.append(
            {
                "sequence_id": sequence_id,
                "rows": len(parsed),
                "path": path.as_posix(),
            }
        )
    report = {
        "valid": not errors,
        "errors": errors,
        "sequences": rows,
        "candidate_name": manifest.get("candidate_name"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "freeze_manifest_sha256": manifest.get("freeze_manifest_sha256"),
    }
    return report


__all__ = [
    "SUBMISSION_COLUMNS",
    "validate_submission_root",
    "write_sequence_results",
    "write_submission_manifest",
]
