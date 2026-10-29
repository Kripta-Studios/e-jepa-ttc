"""Audited, read-only binding from the frozen 8192-token universe to raw eAP events.

The model-facing cache is deliberately separated from TTC supervision.  This module
only handles identity, observation-time, polarity and spatial-coordinate contracts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import hdf5plugin  # noqa: F401  # Register eAP compression filters.
import numpy as np
import pandas as pd

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes

EXPECTED_SEQUENCES = frozenset(
    {
        "2cyv0Oedzg",
        "5ilM1PX2vz",
        "6h5yRW2LGc",
        "OBneIVg4Cw",
        "OYgB6RGWcq",
        "WbCh1DRerJ",
        "mHGFBekt7X",
        "qGsgzl4Q8B",
        "t79dBxj1WS",
    }
)
_H5_KEYS = ("events/x", "events/y", "events/t", "events/p", "ms_to_idx")


def sha256_file(
    path: Path,
    *,
    chunk_bytes: int = 16 * 1024 * 1024,
    resource_check: Callable[[], None] | None = None,
) -> str:
    """Return the complete SHA-256 of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            if resource_check is not None:
                resource_check()
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value using the repository's canonical encoding."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AccessRecord:
    """One physical read recorded by :class:`ReadOnlyTrainAccess`."""

    path: str
    purpose: str
    byte_start: int | None
    byte_stop: int | None
    observed_at_ns: int


class ReadOnlyTrainAccess:
    """Canonical-path allowlist and append-only access ledger for official-train data."""

    def __init__(
        self,
        raw_train_root: Path,
        train_parquet: Path,
        *,
        journal_path: Path | None = None,
        resource_check: Callable[[], None] | None = None,
    ) -> None:
        self.raw_train_root = raw_train_root.resolve(strict=True)
        self.train_parquet = train_parquet.resolve(strict=True)
        if not self.raw_train_root.is_dir() or not self.train_parquet.is_file():
            raise ValueError("raw train root and train parquet must exist")
        self._records: list[AccessRecord] = []
        self.resource_check = resource_check
        self.journal_path = journal_path
        if journal_path is not None:
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            if journal_path.is_file():
                for line in journal_path.read_text(encoding="utf-8").splitlines():
                    self._records.append(AccessRecord(**json.loads(line)))

    @property
    def records(self) -> tuple[AccessRecord, ...]:
        """Return an immutable snapshot of the access ledger."""

        return tuple(self._records)

    def _record(
        self, path: Path, purpose: str, byte_start: int | None = None, byte_stop: int | None = None
    ) -> None:
        if self.resource_check is not None:
            self.resource_check()
        resolved = path.resolve(strict=True)
        allowed = resolved == self.train_parquet or any(
            resolved == (self.raw_train_root / sequence / "events.h5").resolve()
            for sequence in EXPECTED_SEQUENCES
        )
        if not allowed:
            raise ValueError(f"physical read is outside the authorized allowlist: {resolved}")
        record = AccessRecord(str(resolved), purpose, byte_start, byte_stop, time.time_ns())
        if self.journal_path is not None:
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self._records.append(record)

    def parquet_path(self, purpose: str) -> Path:
        """Return the sole authorized parquet path and record the intended read."""

        self._record(self.train_parquet, purpose)
        return self.train_parquet

    def event_path(self, sequence_id: str, source_value: object, purpose: str) -> Path:
        """Resolve an eAP train event path without probing any non-allowlisted location."""

        if sequence_id not in EXPECTED_SEQUENCES:
            raise ValueError(f"sequence is outside the frozen allowlist: {sequence_id}")
        text = str(source_value).replace("\\", "/")
        parts = tuple(part for part in text.split("/") if part not in ("", "."))
        if ".." in parts or Path(text).is_absolute():
            raise ValueError(f"unsafe events path: {source_value!r}")
        expected_suffix = (sequence_id, "events.h5")
        if parts[-2:] != expected_suffix:
            raise ValueError(f"events path does not match sequence {sequence_id}: {source_value!r}")
        candidate = (self.raw_train_root / sequence_id / "events.h5").resolve(strict=True)
        if not candidate.is_relative_to(self.raw_train_root) or not candidate.is_file():
            raise ValueError(f"events path escapes the authorized train root: {candidate}")
        self._record(candidate, purpose)
        return candidate

    def record_h5_slice(self, path: Path, purpose: str, start: int, stop: int) -> None:
        """Record a logical HDF5 event-index range."""

        self._record(path.resolve(strict=True), purpose, int(start), int(stop))

    def write_ledger(self, path: Path) -> None:
        """Persist the access ledger as deterministic CSV."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(AccessRecord.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(item) for item in self._records)

    def audit_summary(self) -> dict[str, Any]:
        """Assess recorded reads without claiming observation of other process I/O."""
        allowed = {str(self.train_parquet)} | {
            str((self.raw_train_root / sequence / "events.h5").resolve())
            for sequence in EXPECTED_SEQUENCES
        }
        violations = sorted({record.path for record in self._records} - allowed)
        return {
            "recorded_reads": len(self._records),
            "recorded_policy_violations": violations,
            "recorded_paths_pass_policy": not violations,
            "durable_journal": self.journal_path is not None,
            "journal_sha256": (
                sha256_file(self.journal_path)
                if self.journal_path is not None and self.journal_path.is_file()
                else None
            ),
            "scope": "reads_requested_through_ReadOnlyTrainAccess",
            "exhaustive_process_access_audit": False,
            "historical_unjournaled_attempts": "not_proven_by_this_journal",
        }


@dataclass(frozen=True)
class RawWindowBinding:
    """One exact half-open event window and its shared spatial transform."""

    sample_token: str
    sequence_id: str
    track_id: str
    outer_fold: int
    window_id: int
    parquet_row_identity: str
    events_path_relative: str
    h5_file_sha256: str
    time_unit: Literal["us"]
    clock_origin_us: int
    frame_to_event_clock_offset_us: int
    frame_to_event_clock_offset_drift_us: int
    window_start_us: int
    window_end_us: int
    observation_end_us: int
    target_anchor_us: int
    target_anchor_event_clock_us: int
    start_event_index: int
    stop_event_index: int
    full_window_event_count: int
    roi_event_count: int | None
    roi_transform_sha256: str
    roi_x0: float
    roi_y0: float
    roi_x1: float
    roi_y1: float
    event_x_offset_px: float
    polarity_encoding: Literal["zero_one", "signed"]
    coordinate_system: str
    read_probe_status: str
    endpoint_delta01_us: int
    endpoint_delta12_us: int


def _reject_external_links(handle: h5py.File, path: Path) -> None:
    for key in _H5_KEYS:
        link = handle.get(key, getlink=True)
        if link is None:
            raise ValueError(f"missing HDF5 object {key!r}: {path}")
        if isinstance(link, h5py.ExternalLink):
            raise ValueError(f"external HDF5 link is forbidden for {key!r}: {path}")
        if not isinstance(handle[key], h5py.Dataset):
            raise ValueError(f"HDF5 object is not a dataset: {key!r}")


def _lower_bound(dataset: h5py.Dataset, value: int, lo: int = 0, hi: int | None = None) -> int:
    stop = len(dataset) if hi is None else int(hi)
    start = int(lo)
    if not 0 <= start <= stop <= len(dataset):
        raise ValueError("invalid HDF5 search bounds")
    while start < stop:
        middle = (start + stop) // 2
        if int(dataset[middle]) < value:
            start = middle + 1
        else:
            stop = middle
    return start


def exact_window_bounds(
    timestamps: h5py.Dataset,
    start_us: int,
    end_us: int,
    *,
    ms_to_idx: np.ndarray,
    index_origin_us: int,
) -> tuple[int, int]:
    """Resolve exact ``[start,end)`` indices, using the coarse map only as a hint."""

    if end_us <= start_us:
        raise ValueError("event window must have positive duration")
    n = len(timestamps)
    coarse = np.asarray(ms_to_idx)
    if coarse.ndim != 1 or not np.issubdtype(coarse.dtype, np.integer):
        raise ValueError("ms_to_idx must be a one-dimensional integer array")
    if np.any(coarse < 0) or np.any(coarse > n) or np.any(coarse[1:] < coarse[:-1]):
        raise ValueError("ms_to_idx values are invalid")
    left_ms = (start_us - index_origin_us) // 1000
    right_ms = (end_us - index_origin_us) // 1000 + 1
    lo = int(coarse[left_ms]) if 0 <= left_ms < len(coarse) else 0
    hi = int(coarse[right_ms]) if 0 <= right_ms < len(coarse) else n
    bracket = (
        lo <= hi
        and (lo == 0 or int(timestamps[lo - 1]) < start_us)
        and (hi == n or int(timestamps[hi]) >= end_us)
    )
    if not bracket:
        lo, hi = 0, n
    first = _lower_bound(timestamps, start_us, lo, hi)
    stop = _lower_bound(timestamps, end_us, first, hi)
    return first, stop


def _as_pairs(value: object, name: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in cast(Any, value):
        pair = np.asarray(item).reshape(-1)
        if pair.shape != (2,):
            raise ValueError(f"{name} entries must contain two values")
        result.append((int(pair[0]), int(pair[1])))
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two entries")
    return result


def _as_boxes(value: object) -> list[tuple[float, float, float, float]]:
    boxes = [tuple(float(x) for x in np.asarray(item).reshape(-1)) for item in cast(Any, value)]
    if len(boxes) != 2 or any(len(item) != 4 for item in boxes):
        raise ValueError("boxes_xyxy must contain exactly two boxes")
    return cast(list[tuple[float, float, float, float]], boxes)


def _as_two_timestamps(value: object) -> tuple[int, int]:
    values = np.asarray(value).reshape(-1)
    if values.shape != (2,):
        raise ValueError("frame_timestamps_us must contain exactly two entries")
    first, second = int(values[0]), int(values[1])
    if second <= first:
        raise ValueError("frame timestamps must be strictly increasing")
    return first, second


def _polarity_encoding(dataset: h5py.Dataset) -> Literal["zero_one", "signed"]:
    values: set[int] = set()
    length = len(dataset)
    for start in sorted({0, max(0, length // 2 - 2048), max(0, length - 4096)}):
        values.update(np.asarray(dataset[start : min(length, start + 4096)]).astype(int).tolist())
    if values <= {0, 1} and values:
        return "zero_one"
    if values <= {-1, 1} and values:
        return "signed"
    raise ValueError(f"unsupported or ambiguous polarity values: {sorted(values)}")


def _stable_file_hashes(
    event_paths: dict[str, Path],
    progress_path: Path | None = None,
    resource_check: Callable[[], None] | None = None,
) -> dict[str, dict[str, Any]]:
    progress: dict[str, dict[str, Any]] = {}
    if progress_path is not None and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    for sequence_id, path in sorted(event_paths.items()):
        if resource_check is not None:
            resource_check()
        before = path.stat()
        prior = progress.get(sequence_id)
        if (
            prior
            and prior.get("bytes") == before.st_size
            and prior.get("mtime_ns") == before.st_mtime_ns
        ):
            continue
        digest = sha256_file(path, resource_check=resource_check)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"HDF5 changed while hashing: {path}")
        progress[sequence_id] = {
            "path": str(path),
            "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "sha256": digest,
        }
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, progress_path)
    return progress


def build_raw_window_bindings(
    *,
    stage_metadata_path: Path,
    train_parquet: Path,
    raw_train_root: Path,
    access: ReadOnlyTrainAccess,
    hash_progress_path: Path | None = None,
    expected_tokens: int = 8192,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build all 16,384 physical window bindings for the frozen universe."""

    stage = pd.read_csv(stage_metadata_path, dtype=str)
    required_stage = {"sample_token", "sequence_id", "track_id", "outer_fold"}
    if not required_stage <= set(stage) or len(stage) != expected_tokens:
        raise ValueError("stage metadata does not match the frozen token contract")
    if stage["sample_token"].duplicated().any() or set(stage["sequence_id"]) != EXPECTED_SEQUENCES:
        raise ValueError("stage token identity or nine-sequence allowlist mismatch")
    source_columns = [
        "sample_token",
        "sequence_id",
        "track_id",
        "timestamp_us",
        "frame_timestamps_us",
        "events_path",
        "event_windows_us",
        "boxes_xyxy",
    ]
    source = pd.read_parquet(access.parquet_path("raw_binding_join"), columns=source_columns)
    if source["sample_token"].duplicated().any():
        raise ValueError("train parquet contains duplicate sample_token values")
    merged = stage.merge(
        source, on=["sample_token", "sequence_id", "track_id"], validate="one_to_one"
    )
    if len(merged) != expected_tokens:
        raise ValueError("token-to-parquet join is not complete and one-to-one")
    event_paths: dict[str, Path] = {}
    for row in merged.itertuples(index=False):
        row = cast(Any, row)
        sequence = str(row.sequence_id)
        path = access.event_path(sequence, row.events_path, "resolve_raw_hdf5")
        prior = event_paths.setdefault(sequence, path)
        if prior != path:
            raise ValueError(f"sequence maps to multiple HDF5 files: {sequence}")
    hashes = _stable_file_hashes(event_paths, hash_progress_path, access.resource_check)

    file_info: dict[str, dict[str, Any]] = {}
    handles: dict[str, h5py.File] = {}
    try:
        for sequence, path in sorted(event_paths.items()):
            handle = h5py.File(path, "r")
            handles[sequence] = handle
            _reject_external_links(handle, path)
            lengths = {key: len(cast(h5py.Dataset, handle[key])) for key in _H5_KEYS[:4]}
            if len(set(lengths.values())) != 1 or next(iter(lengths.values())) == 0:
                raise ValueError(f"event fields are empty or length-mismatched: {path}")
            timestamps = cast(h5py.Dataset, handle["events/t"])
            probes = np.asarray(timestamps[:: max(1, len(timestamps) // 4096)], dtype=np.int64)
            if np.any(np.diff(probes) < 0):
                raise ValueError(f"sampled timestamps are not monotonic: {path}")
            coarse = np.asarray(cast(h5py.Dataset, handle["ms_to_idx"]), dtype=np.int64)
            first_us = int(timestamps[0])
            # eAP ms_to_idx is indexed by absolute stored milliseconds in these files.
            origin = 0 if len(coarse) > first_us // 1000 else (first_us // 1000) * 1000
            file_info[sequence] = {
                "timestamps": timestamps,
                "coarse": coarse,
                "origin": origin,
                "polarity": _polarity_encoding(cast(h5py.Dataset, handle["events/p"])),
                "event_count": lengths["events/t"],
            }

        rows: list[dict[str, Any]] = []
        clock_offsets: dict[str, set[int]] = {sequence: set() for sequence in event_paths}
        for parquet_row, row in enumerate(merged.itertuples(index=False)):
            row = cast(Any, row)
            sequence = str(row.sequence_id)
            windows = _as_pairs(row.event_windows_us, "event_windows_us")
            frames = _as_two_timestamps(row.frame_timestamps_us)
            boxes = _as_boxes(row.boxes_xyxy)
            square = common_square_from_boxes(boxes, (0, 1), margin_fraction=0.25, minimum_edge=8.0)
            roi_contract = {
                "square_xyxy": list(square),
                "canvas": [64, 64],
                "event_x_offset_px": 5.0,
                "projection": "hard_floor_half_open",
                "source": "common_square_from_boxes_v4_t1_t2_margin025_min8",
            }
            roi_sha = canonical_json_sha256(roi_contract)
            info = file_info[sequence]
            timestamp_data = cast(h5py.Dataset, info["timestamps"])
            frame_offset_values = [frames[i] - windows[i][1] for i in range(2)]
            clock_offset_drift = frame_offset_values[1] - frame_offset_values[0]
            if abs(clock_offset_drift) > 5:
                raise ValueError(
                    f"frame/event clock drift exceeds 5 us for {row.sample_token}: "
                    f"{clock_offset_drift}"
                )
            clock_offsets[sequence].update(frame_offset_values)
            precontext_end = windows[0][1] - max(100_000, windows[0][1] - windows[0][0])
            delta01 = windows[0][1] - precontext_end
            delta12 = frames[1] - frames[0]
            for window_id, (start_us, end_us) in enumerate(windows):
                start_idx, stop_idx = exact_window_bounds(
                    timestamp_data,
                    start_us,
                    end_us,
                    ms_to_idx=cast(np.ndarray, info["coarse"]),
                    index_origin_us=int(info["origin"]),
                )
                if stop_idx <= start_idx:
                    raise ValueError(f"empty raw window for token {row.sample_token}")
                if start_idx and int(timestamp_data[start_idx - 1]) >= start_us:
                    raise AssertionError("left boundary proof failed")
                if stop_idx < len(timestamp_data) and int(timestamp_data[stop_idx]) < end_us:
                    raise AssertionError("right boundary proof failed")
                binding = RawWindowBinding(
                    sample_token=str(row.sample_token),
                    sequence_id=sequence,
                    track_id=str(row.track_id),
                    outer_fold=int(row.outer_fold),
                    window_id=window_id,
                    parquet_row_identity=f"row:{parquet_row}:{row.sample_token}",
                    events_path_relative=f"{sequence}/events.h5",
                    h5_file_sha256=str(hashes[sequence]["sha256"]),
                    time_unit="us",
                    clock_origin_us=int(info["origin"]),
                    frame_to_event_clock_offset_us=frame_offset_values[window_id],
                    frame_to_event_clock_offset_drift_us=clock_offset_drift,
                    window_start_us=start_us,
                    window_end_us=end_us,
                    observation_end_us=end_us,
                    target_anchor_us=frames[1],
                    target_anchor_event_clock_us=windows[1][1],
                    start_event_index=start_idx,
                    stop_event_index=stop_idx,
                    full_window_event_count=stop_idx - start_idx,
                    roi_event_count=None,
                    roi_transform_sha256=roi_sha,
                    roi_x0=square[0],
                    roi_y0=square[1],
                    roi_x1=square[2],
                    roi_y1=square[3],
                    event_x_offset_px=5.0,
                    polarity_encoding=cast(Literal["zero_one", "signed"], info["polarity"]),
                    coordinate_system="eap_event_sensor_to_rgb_x_plus_5_then_common_roi_64",
                    read_probe_status="pending",
                    endpoint_delta01_us=delta01,
                    endpoint_delta12_us=delta12,
                )
                rows.append(asdict(binding))
    finally:
        for handle in handles.values():
            handle.close()
    frame = pd.DataFrame(rows).sort_values(["sample_token", "window_id"]).reset_index(drop=True)
    if len(frame) != 2 * expected_tokens or frame.duplicated(["sample_token", "window_id"]).any():
        raise AssertionError("raw window binding did not produce two unique rows per token")
    manifest = {
        "artifact_type": "scientific_recovery_v9_stage63_raw_binding_v2",
        "rows": len(frame),
        "tokens": int(cast(Any, frame["sample_token"]).nunique()),
        "sequences": int(cast(Any, frame["sequence_id"]).nunique()),
        "interval": "half_open_start_inclusive_end_exclusive",
        "hdf5_files": hashes,
        "clock_offsets_by_sequence": {
            key: sorted(value) for key, value in sorted(clock_offsets.items())
        },
        "within_token_clock_drift": {
            "definition": "offset_window1_minus_offset_window0_us",
            "allowed_absolute_max_us": 5,
            "observed_min_us": int(frame["frame_to_event_clock_offset_drift_us"].min()),
            "observed_max_us": int(frame["frame_to_event_clock_offset_drift_us"].max()),
        },
        "common_roi": {
            "same_transform_for_both_windows": bool(
                np.asarray(
                    frame.groupby("sample_token")["roi_transform_sha256"].nunique() == 1
                ).all()
            ),
            "margin_fraction": 0.25,
            "minimum_edge_px": 8.0,
            "canvas": [64, 64],
            "event_x_offset_px": 5.0,
        },
        "forbidden_paths_opened": None,
        "recorded_access_assessment": access.audit_summary(),
        "targets_in_binding": False,
    }
    return frame, manifest


def select_hash_probe_tokens(binding: pd.DataFrame, count: int = 64) -> list[str]:
    """Select a deterministic hash-ranked, sequence-stratified physical-read probe."""

    tokens = binding[["sample_token", "sequence_id"]].drop_duplicates()
    if count < len(EXPECTED_SEQUENCES) or count > len(tokens):
        raise ValueError("invalid physical probe size")
    base, remainder = divmod(count, len(EXPECTED_SEQUENCES))
    result: list[str] = []
    for index, sequence in enumerate(sorted(EXPECTED_SEQUENCES)):
        group = cast(
            pd.Series,
            tokens.loc[np.asarray(tokens["sequence_id"] == sequence), "sample_token"],
        ).astype(str)
        ranked = sorted(group, key=lambda token: hashlib.sha256(token.encode()).digest())
        result.extend(ranked[: base + (index < remainder)])
    if len(result) != count or len(set(result)) != count:
        raise AssertionError("hash probe is not unique and complete")
    return result


__all__ = [
    "EXPECTED_SEQUENCES",
    "RawWindowBinding",
    "ReadOnlyTrainAccess",
    "build_raw_window_bindings",
    "canonical_json_sha256",
    "exact_window_bounds",
    "select_hash_probe_tokens",
    "sha256_file",
]
