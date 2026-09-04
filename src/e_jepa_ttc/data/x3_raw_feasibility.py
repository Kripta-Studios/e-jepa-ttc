"""Read-only raw-event binding and feasibility utilities for Stage 61 X3."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import h5py
import hdf5plugin  # noqa: F401  # Registers the filters used by the eAP HDF5 files.
import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import sign_artifact

FORBIDDEN_PATH_PARTS = frozenset({"public", "private", "test", "codabench"})
RAW_FIELDS = ("x", "y", "t", "p")


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a physical file with a throughput-oriented block size."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_relative_path(value: object) -> PurePosixPath:
    relative = PurePosixPath(str(value).replace("\\", "/"))
    lowered = {part.lower() for part in relative.parts}
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"event path must be a safe relative path: {value!r}")
    if forbidden := sorted(lowered & FORBIDDEN_PATH_PARTS):
        raise ValueError(f"event path enters a sealed partition {forbidden}: {value!r}")
    if "train" not in lowered:
        raise ValueError(f"event path is not explicitly train-bound: {value!r}")
    return relative


def _resolve_train_event_path(eap_root: Path, value: object) -> tuple[str, Path]:
    relative = _normalise_relative_path(value)
    root = eap_root.resolve(strict=True)
    resolved = (root / Path(*relative.parts)).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"event path escapes eAP root: {value!r}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative.as_posix(), resolved


def _window_pairs(value: object) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_pair in value:  # type: ignore[union-attr]
        pair = np.asarray(raw_pair).tolist()
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"invalid event window: {raw_pair!r}")
        start_us, end_us = (int(item) for item in pair)
        if start_us < 0 or end_us <= start_us:
            raise ValueError(f"non-positive event window: {(start_us, end_us)}")
        pairs.append((start_us, end_us))
    if not pairs:
        raise ValueError("sample token has no event windows")
    return pairs


def _exact_timestamp_index(
    timestamps: h5py.Dataset,
    ms_to_idx: np.ndarray,
    timestamp_us: int,
    cache: dict[int, int],
) -> int:
    cached = cache.get(timestamp_us)
    if cached is not None:
        return cached
    millisecond = timestamp_us // 1000
    if millisecond < 0 or millisecond >= len(ms_to_idx):
        raise IndexError(f"timestamp {timestamp_us} is outside ms_to_idx")
    coarse_start = int(ms_to_idx[millisecond])
    coarse_end = (
        int(ms_to_idx[millisecond + 1]) if millisecond + 1 < len(ms_to_idx) else len(timestamps)
    )
    if coarse_end < coarse_start:
        raise ValueError("ms_to_idx is not monotonic at timestamp boundary")
    values = np.asarray(timestamps[coarse_start:coarse_end], dtype=np.int64)
    exact = coarse_start + int(np.searchsorted(values, timestamp_us, side="left"))
    cache[timestamp_us] = exact
    return exact


def _sample_polarities(dataset: h5py.Dataset) -> list[int]:
    length = len(dataset)
    values: set[int] = set()
    for centre in sorted({0, length // 2, max(0, length - 1)}):
        start = max(0, centre - 128)
        end = min(length, centre + 129)
        values.update(int(item) for item in np.asarray(dataset[start:end]).tolist())
    return sorted(values)


def _select_probe_rows(binding: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or count > len(binding):
        raise ValueError(f"read probe count must be in [1, {len(binding)}], got {count}")
    groups = list(binding.sort_values(["sequence_id", "sample_token"]).groupby("sequence_id"))
    base, remainder = divmod(count, len(groups))
    selected: list[pd.DataFrame] = []
    for index, (_, group) in enumerate(groups):
        take = base + (1 if index < remainder else 0)
        if take > len(group):
            raise ValueError("not enough tokens for a stratified read probe")
        positions = np.linspace(0, len(group) - 1, take, dtype=int)
        selected.append(group.iloc[positions])
    result = pd.concat(selected, ignore_index=True)
    if len(result) != count or result["sample_token"].nunique() != count:
        raise AssertionError("read probe selection is not token-unique")
    return result


def _raw_read_probe(binding: pd.DataFrame, *, count: int) -> dict[str, Any]:
    selected = _select_probe_rows(binding, count)
    checksum = hashlib.sha256()
    logical_bytes = 0
    events = 0
    began = time.perf_counter()
    per_sequence: dict[str, dict[str, int]] = {}
    for event_path, rows in selected.groupby("resolved_events_path", sort=True):
        with h5py.File(str(event_path), "r") as handle:
            event_group = cast(h5py.Group, handle["events"])
            sequence_events = 0
            sequence_bytes = 0
            for row in _frame_records(rows):
                for window_index in range(int(row["window_count"])):
                    start_index = int(row[f"window{window_index}_event_start_idx"])
                    end_index = int(row[f"window{window_index}_event_end_idx"])
                    window_events = end_index - start_index
                    events += window_events
                    sequence_events += window_events
                    for field in RAW_FIELDS:
                        dataset = cast(h5py.Dataset, event_group[field])
                        values = np.asarray(dataset[start_index:end_index])
                        logical_bytes += values.nbytes
                        sequence_bytes += values.nbytes
                        if len(values):
                            checksum.update(values[:1].tobytes())
                            checksum.update(values[-1:].tobytes())
            sequence_id = str(rows["sequence_id"].iloc[0])
            per_sequence[sequence_id] = {
                "tokens": len(rows),
                "events": sequence_events,
                "logical_bytes": sequence_bytes,
            }
    elapsed = time.perf_counter() - began
    return {
        "selection": "deterministic_stratified_evenly_spaced",
        "tokens": len(selected),
        "tokens_unique": len(set(selected["sample_token"].astype(str).tolist())),
        "sequences": len(set(selected["sequence_id"].astype(str).tolist())),
        "windows": int(np.asarray(selected["window_count"], dtype=np.int64).sum()),
        "events": events,
        "logical_bytes": logical_bytes,
        "seconds": elapsed,
        "tokens_per_second": len(selected) / max(elapsed, 1e-12),
        "events_per_second": events / max(elapsed, 1e-12),
        "logical_mib_per_second": logical_bytes / (1024**2) / max(elapsed, 1e-12),
        "probe_checksum": checksum.hexdigest(),
        "per_sequence": per_sequence,
    }


def _cache_proposal(binding: pd.DataFrame) -> dict[str, Any]:
    total_duration_us = 0
    total_microbins = 0
    total_snapshots = 0
    for row in _frame_records(binding):
        for window_index in range(int(row["window_count"])):
            start_us = int(row[f"window{window_index}_start_us"])
            end_us = int(row[f"window{window_index}_end_us"])
            duration_us = end_us - start_us
            total_duration_us += duration_us
            total_microbins += math.ceil(duration_us / 1000)
            total_snapshots += math.ceil(duration_us / 5000)
    total_events = int(np.asarray(binding["raw_event_count"], dtype=np.int64).sum())
    return {
        "microbin_us": 1000,
        "snapshot_interval_us": 5000,
        "raw_fields": ["x", "y", "timestamp_us", "polarity", "sample_token"],
        "tokens": len(binding),
        "windows": int(np.asarray(binding["window_count"], dtype=np.int64).sum()),
        "total_duration_us_with_overlap": total_duration_us,
        "total_event_references_with_overlap": total_events,
        "naive_per_token_raw_bytes_uncompressed": total_events * 13,
        "naive_per_token_raw_layout": {
            "x_uint16": 2,
            "y_uint16": 2,
            "timestamp_int64": 8,
            "polarity_int8": 1,
        },
        "recommended_storage": "zero-copy HDF5 ranges plus signed token binding",
        "microbin_count": total_microbins,
        "microbin_patch_polarity_count_uint32_bytes": total_microbins * 16 * 2 * 4,
        "snapshot_count": total_snapshots,
        "snapshot_patch34_float32_bytes": total_snapshots * 16 * 34 * 4,
        "estimate_scope": "storage-only estimate; no X3 model architecture or training authorized",
    }


def build_x3_raw_binding(
    *,
    stage_metadata_path: Path,
    garl_train_parquet: Path,
    eap_root: Path,
    binding_csv_path: Path,
    binding_manifest_path: Path,
    code_commit: str,
    protocol_sha256: str,
    expected_tokens: int = 8192,
    read_probe_tokens: int = 64,
    hash_event_files: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build and validate a signed zero-copy binding from tokens to train raw events."""

    stage = pd.read_csv(
        stage_metadata_path,
        usecols=["sample_token", "sequence_id", "track_id"],
        dtype=str,
    )
    if len(stage) != expected_tokens or stage["sample_token"].nunique() != expected_tokens:
        raise ValueError(
            f"Stage metadata must contain {expected_tokens} unique tokens, got "
            f"rows={len(stage)} unique={stage['sample_token'].nunique()}"
        )
    source_columns = [
        "sample_token",
        "sequence_id",
        "track_id",
        "events_path",
        "event_windows_us",
    ]
    source = pd.read_parquet(garl_train_parquet, columns=source_columns)
    if bool(source["sample_token"].duplicated().to_numpy().any()):
        raise ValueError("GarlTTC train manifest contains duplicate sample_token values")
    merged = stage.merge(
        source,
        on="sample_token",
        how="left",
        suffixes=("_stage", "_source"),
        validate="one_to_one",
        indicator=True,
    )
    if not bool(merged["_merge"].eq("both").to_numpy().all()):
        missing = merged.loc[merged["_merge"].ne("both"), "sample_token"].tolist()
        raise ValueError(f"raw binding missing {len(missing)} tokens; first={missing[:3]}")
    if not bool(merged["sequence_id_stage"].eq(merged["sequence_id_source"]).to_numpy().all()):
        raise ValueError("sequence_id mismatch between Stage metadata and GarlTTC manifest")
    if not bool(merged["track_id_stage"].eq(merged["track_id_source"]).to_numpy().all()):
        raise ValueError("track_id mismatch between Stage metadata and GarlTTC manifest")

    merged["sequence_id"] = merged.pop("sequence_id_stage")
    merged["track_id"] = merged.pop("track_id_stage")
    binding_rows: list[dict[str, Any]] = []
    event_file_records: list[dict[str, Any]] = []
    for raw_path in sorted(merged["events_path"].unique()):
        relative, resolved = _resolve_train_event_path(eap_root, raw_path)
        rows = merged.loc[merged["events_path"].eq(raw_path)]
        with h5py.File(resolved, "r") as handle:
            if "events" not in handle or "ms_to_idx" not in handle:
                raise ValueError(f"missing events/ms_to_idx in {resolved}")
            event_group = cast(h5py.Group, handle["events"])
            missing_fields = sorted(set(RAW_FIELDS) - set(event_group.keys()))
            if missing_fields:
                raise ValueError(f"missing raw fields {missing_fields} in {resolved}")
            datasets = {field: cast(h5py.Dataset, event_group[field]) for field in RAW_FIELDS}
            lengths = {field: len(datasets[field]) for field in RAW_FIELDS}
            if len(set(lengths.values())) != 1 or next(iter(lengths.values())) <= 0:
                raise ValueError(f"raw field length mismatch in {resolved}: {lengths}")
            timestamps = datasets["t"]
            ms_to_idx = np.asarray(cast(h5py.Dataset, handle["ms_to_idx"]), dtype=np.int64)
            if len(ms_to_idx) == 0 or np.any(ms_to_idx[1:] < ms_to_idx[:-1]):
                raise ValueError(f"invalid ms_to_idx in {resolved}")
            boundary_cache: dict[int, int] = {}
            file_start_us = int(timestamps[0])
            file_end_us = int(timestamps[-1])
            for row in _frame_records(rows):
                pairs = _window_pairs(row["event_windows_us"])
                record: dict[str, Any] = {
                    "sample_token": str(row["sample_token"]),
                    "sequence_id": str(row["sequence_id"]),
                    "track_id": str(row["track_id"]),
                    "events_relpath": relative,
                    "resolved_events_path": str(resolved),
                    "window_count": len(pairs),
                    "raw_interval_start_us": pairs[0][0],
                    "raw_interval_end_us": pairs[-1][1],
                }
                total_count = 0
                for window_index, (start_us, end_us) in enumerate(pairs):
                    if start_us < file_start_us or end_us > file_end_us:
                        raise ValueError(
                            f"token {row['sample_token']} window {(start_us, end_us)} "
                            f"outside HDF5 range {(file_start_us, file_end_us)}"
                        )
                    start_index = _exact_timestamp_index(
                        timestamps, ms_to_idx, start_us, boundary_cache
                    )
                    end_index = _exact_timestamp_index(
                        timestamps, ms_to_idx, end_us, boundary_cache
                    )
                    count = end_index - start_index
                    if count <= 0:
                        raise ValueError(f"empty raw window for token {row['sample_token']}")
                    record.update(
                        {
                            f"window{window_index}_start_us": start_us,
                            f"window{window_index}_end_us": end_us,
                            f"window{window_index}_event_start_idx": start_index,
                            f"window{window_index}_event_end_idx": end_index,
                            f"window{window_index}_event_count": count,
                        }
                    )
                    total_count += count
                record["raw_event_count"] = total_count
                binding_rows.append(record)
            event_file_records.append(
                {
                    "sequence_id": str(rows["sequence_id"].iloc[0]),
                    "relative_path": relative,
                    "resolved_path": str(resolved),
                    "bytes": resolved.stat().st_size,
                    "sha256": None,
                    "event_count": lengths["t"],
                    "t_first_us": file_start_us,
                    "t_last_us": file_end_us,
                    "ms_to_idx_entries": len(ms_to_idx),
                    "raw_fields": {
                        field: {
                            "dtype": str(datasets[field].dtype),
                            "shape": list(datasets[field].shape),
                        }
                        for field in RAW_FIELDS
                    },
                    "sampled_polarity_values": _sample_polarities(datasets["p"]),
                }
            )

    binding = pd.DataFrame(binding_rows).sort_values("sample_token").reset_index(drop=True)
    if len(binding) != expected_tokens or binding["sample_token"].nunique() != expected_tokens:
        raise AssertionError("constructed raw binding lost or duplicated tokens")
    probe = _raw_read_probe(binding, count=read_probe_tokens)

    if hash_event_files:
        for item in event_file_records:
            item["sha256"] = sha256_file(Path(str(item["resolved_path"])))
    event_hash_by_path = {str(item["relative_path"]): item["sha256"] for item in event_file_records}
    binding["events_file_sha256"] = binding["events_relpath"].apply(
        lambda value: event_hash_by_path[str(value)]
    )
    binding_csv_path.parent.mkdir(parents=True, exist_ok=True)
    binding.to_csv(binding_csv_path, index=False, lineterminator="\n")
    binding_sha256 = sha256_file(binding_csv_path)
    source_sha256 = sha256_file(garl_train_parquet)
    stage_metadata_sha256 = sha256_file(stage_metadata_path)
    proposal = _cache_proposal(binding)
    manifest = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v9_x3_raw_binding_v1",
            "schema_version": "1.0",
            "evidence_type": "read_only_feasibility",
            "code_commit": code_commit,
            "protocol_version": "stage61_stage62_v1",
            "protocol_sha256": protocol_sha256,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "completed",
            "training_executed": False,
            "binding_csv": {
                "path": str(binding_csv_path),
                "sha256": binding_sha256,
                "rows": len(binding),
            },
            "stage_metadata": {
                "path": str(stage_metadata_path),
                "sha256": stage_metadata_sha256,
            },
            "garl_train_manifest": {
                "path": str(garl_train_parquet),
                "sha256": source_sha256,
                "rows": len(source),
            },
            "eap_root": str(eap_root.resolve(strict=True)),
            "tokens": len(binding),
            "tokens_unique": len(set(binding["sample_token"].astype(str).tolist())),
            "sequences": len(set(binding["sequence_id"].astype(str).tolist())),
            "tracks": len(set(binding["track_id"].astype(str).tolist())),
            "windows": int(np.asarray(binding["window_count"], dtype=np.int64).sum()),
            "raw_event_references": int(
                np.asarray(binding["raw_event_count"], dtype=np.int64).sum()
            ),
            "event_files": event_file_records,
            "forbidden_paths_opened": False,
            "labels_opened": False,
            "all_paths_train_only": True,
            "timestamps_available": True,
            "polarity_available": True,
            "event_counts_per_token_available": True,
            "raw_intervals_per_token_available": True,
            "read_probe": probe,
            "future_cache_proposal": proposal,
        }
    )
    binding_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    binding_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest, probe, proposal
