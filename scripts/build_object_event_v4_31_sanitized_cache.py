#!/usr/bin/env python3
"""Materialize the label-free v4.31 event cache from the allowed train parquet only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from e_jepa_ttc.data.event_v4_common_roi import trajectory_common_roi  # noqa: E402, I001
from e_jepa_ttc.data.object_event_v4_31 import (  # noqa: E402
    PROJECTED_COLUMNS,
    SPLIT_CONTRACT,
    SPLIT_PATH,
    SOURCE_SHA256,
    AtomicDirectory,
    reject_forbidden_path,
    resolve_event_path,
    load_split_contract,
    sanitize_row,
    scientific_metadata,
    select_split,
    sha256_file,
    strict_json,
    validate_projection,
)
from e_jepa_ttc.data.event_v4_geometry import shifted_precontext_window  # noqa: E402


class EventH5Reader:
    """Persistent index-assisted reader; no full timestamp array is materialized."""

    def __init__(self, path: Path) -> None:
        self.handle = h5py.File(path, "r")
        self.group = cast(
            h5py.Group, self.handle["events"] if "events" in self.handle else self.handle
        )
        self.index = cast(h5py.Dataset, self.handle["ms_to_idx"])
        if self.index.ndim != 1 or len(self.index) < 2:
            self.close()
            raise ValueError("ms_to_idx must be a one-dimensional nontrivial dataset")
        missing = [key for key in ("x", "y", "t", "p") if key not in self.group]
        if missing:
            self.close()
            raise ValueError(f"events HDF5 group lacks datasets: {missing}")
        lengths = {len(cast(h5py.Dataset, self.group[key])) for key in ("x", "y", "t", "p")}
        if len(lengths) != 1:
            self.close()
            raise ValueError("event coordinate/timestamp/polarity arrays have different lengths")

    def read(
        self, intervals: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> dict[str, np.ndarray]:
        lower, upper = min(x[0] for x in intervals), max(x[1] for x in intervals)
        if lower < 0 or upper <= lower:
            raise ValueError("absolute event interval bounds must be positive and ordered")
        start_ms, end_ms = lower // 1000, (upper + 999) // 1000
        if start_ms < 0 or end_ms >= len(self.index):
            raise ValueError("requested event interval falls outside ms_to_idx bounds")
        left, right = int(self.index[start_ms]), int(self.index[end_ms])
        total = len(cast(h5py.Dataset, self.group["t"]))
        if not 0 <= left <= right <= total:
            raise ValueError("ms_to_idx contains invalid event slice bounds")
        timestamps = np.asarray(cast(h5py.Dataset, self.group["t"])[left:right])
        if timestamps.ndim != 1 or (len(timestamps) > 1 and np.any(np.diff(timestamps) < 0)):
            raise ValueError("bounded event timestamps must be monotonic")
        begin = left + int(np.searchsorted(timestamps, lower, side="left"))
        finish = left + int(np.searchsorted(timestamps, upper, side="left"))
        result = {
            key: np.asarray(cast(h5py.Dataset, self.group[key])[begin:finish])
            for key in ("x", "y", "t", "p")
        }
        polarity = result["p"]
        if not np.isin(polarity, (-1, 0, 1)).all():
            raise ValueError("bounded event polarity must be drawn from {-1,0,+1}")
        return result

    def close(self) -> None:
        self.handle.close()


def _close_memmap(array: np.memmap) -> None:
    """Flush and release a NumPy mapping before an atomic directory rename."""
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is None:
        raise RuntimeError("NumPy memmap does not expose its backing mapping")
    mapping.close()


def _jsonl_record(value: object) -> str:
    """Serialize one finite, deterministic JSON object on exactly one line."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _intervals(value: object) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Normalize the projected three absolute event intervals without inferring data."""
    raw = np.asarray(value, dtype=np.int64)
    if raw.shape != (2, 2):
        raise ValueError(f"event_windows_us must be [2,2] t1/t2 intervals, got {raw.shape}")
    t1 = (int(raw[0, 0]), int(raw[0, 1]))
    t2 = (int(raw[1, 0]), int(raw[1, 1]))
    if t1[0] < 100_000 or t1[0] >= t1[1] or t2[0] >= t2[1] or t1[1] > t2[0]:
        raise ValueError("t1/t2 windows must be positive, ordered, and non-overlapping")
    t0 = shifted_precontext_window(t1, shift_s=0.1)
    if (
        t0[1] - t0[0] != t1[1] - t1[0]
        or t1[0] - t0[0] < 100_000
        or t0[1] > t1[0]
    ):
        raise RuntimeError("unexpected shifted t0 pre-context convention")
    return t0, t1, t2


def _boxes(value: object) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(value, dtype=np.float64)
    # Rows may carry [3,4] boxes; common ROI only consumes t1/t2, t0 is proxy t1.
    if raw.shape == (3, 4):
        return raw[1], raw[2]
    if raw.shape == (2, 4):
        return raw[0], raw[1]
    raise ValueError(f"boxes_xyxy must be [2,4] or [3,4], got {raw.shape}")


def _read_events(
    path: Path,
    intervals: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    readers: dict[Path, EventH5Reader],
) -> dict[str, np.ndarray]:
    if path not in readers:
        readers[path] = EventH5Reader(path)
    return readers[path].read(intervals)


def _trajectory(
    row: dict[str, Any], readers: dict[Path, EventH5Reader], event_root: Path
) -> np.ndarray:
    intervals = _intervals(row["event_windows_us"])
    box_t1, box_t2 = _boxes(row["boxes_xyxy"])
    path = resolve_event_path(str(row["events_path"]), event_root=event_root)
    events = _read_events(path, intervals, readers)
    return (
        trajectory_common_roi(
            events,
            t0=intervals[0],
            t1=intervals[1],
            t2=intervals[2],
            box_t1=cast(list[float], box_t1.tolist()),
            box_t2=cast(list[float], box_t2.tolist()),
        )
        .numpy()
        .astype(np.float16, copy=False)
    )


def build(
    config: dict[str, Any],
    output: Path,
    *,
    full: bool,
    force: bool = False,
    event_root: Path | None = None,
) -> dict[str, Any]:
    source_value = str(config["source"]["train_parquet"])
    source = Path(os.path.expandvars(source_value))
    if "${" in str(source):
        raise ValueError("source parquet environment template was not resolved")
    reject_forbidden_path(source, source=True)
    validate_projection(PROJECTED_COLUMNS)
    split_path = Path(os.path.expandvars(str(config["split"])))
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if split_path.resolve() != SPLIT_PATH.resolve():
        raise ValueError("v4.31 config must bind the authoritative split path")
    split_sha = sha256_file(split_path)
    if config.get("split_sha256") != split_sha:
        raise ValueError("v4.31 split SHA differs from config")
    load_split_contract(split_path)
    expected = config["source"].get("sha256", SOURCE_SHA256)
    if expected != SOURCE_SHA256:
        raise ValueError("v4.31 source SHA must equal the locked allowed parquet SHA")
    if sha256_file(source) != expected:
        raise ValueError("train parquet SHA256 mismatch")
    table = pq.read_table(source, columns=list(PROJECTED_COLUMNS))
    rows = table.to_pylist()
    selected = select_split(rows, full=full)
    root_value = os.path.expandvars(str(config.get("event_root", "")))
    resolved_event_root = event_root or Path(os.environ.get("E_JEPA_TTC_EVENT_ROOT", root_value))
    if not str(resolved_event_root) or "${" in str(resolved_event_root):
        raise ValueError("event root must be passed or resolved from E_JEPA_TTC_EVENT_ROOT")
    effective_config = dict(config)
    effective_source = dict(config["source"])
    effective_source["train_parquet"] = str(source.resolve())
    effective_config["source"] = effective_source
    effective_config["event_root"] = str(resolved_event_root.resolve())
    effective_config["split"] = str(split_path.resolve())
    config_identity = hashlib.sha256(strict_json(effective_config).encode("utf-8")).hexdigest()
    source_identity = f"{source.resolve()}:{expected}"
    with AtomicDirectory(
        output,
        force=force,
        config_identity=config_identity,
        source_identity=source_identity,
    ) as stage:
        events = np.lib.format.open_memmap(
            stage / "events.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(selected), 3, 12, 128, 128),
        )
        deltas = np.lib.format.open_memmap(
            stage / "delta_t_s.npy", mode="w+", dtype=np.float32, shape=(len(selected),)
        )
        readers: dict[Path, EventH5Reader] = {}
        try:
            for index, row in enumerate(selected):
                events[index] = _trajectory(row, readers, resolved_event_root)
                deltas[index] = sanitize_row(row, index)["delta_t_s"]
        finally:
            for reader in readers.values():
                reader.close()
            event_shape = list(events.shape)
            # An open mapping prevents AtomicDirectory from promoting or cleaning
            # its staging directory on Windows.  Always release both handles,
            # including when materialization itself raises.
            _close_memmap(events)
            _close_memmap(deltas)
        with (stage / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(selected):
                handle.write(_jsonl_record(sanitize_row(row, index)) + "\n")
        manifest = {
            "artifact_type": "object_event_v4_31_sanitized_cache_v1",
            **scientific_metadata(
                artifact_type="object_event_v4_31_sanitized_cache_v1",
                evidence_type="sanitized_event_roi_cache",
                protocol_version=SPLIT_CONTRACT["version"],
                protocol_sha256=split_sha,
                artifact_sha256=hashlib.sha256(
                    strict_json(
                        {
                            "events": sha256_file(stage / "events.npy"),
                            "delta_t_s": sha256_file(stage / "delta_t_s.npy"),
                            "rows": sha256_file(stage / "rows.jsonl"),
                        }
                    ).encode("utf-8")
                ).hexdigest(),
            ),
            "mode": "full" if full else "diagnostic",
            "count": len(selected),
            "events": {
                "path": "events.npy",
                "dtype": "float16",
                "shape": event_shape,
                "sha256": sha256_file(stage / "events.npy"),
            },
            "delta_t_s": {
                "path": "delta_t_s.npy",
                "dtype": "float32",
                "sha256": sha256_file(stage / "delta_t_s.npy"),
            },
            "rows_path": "rows.jsonl",
            "rows_sha256": sha256_file(stage / "rows.jsonl"),
            "source": {
                "path": str(source.resolve()),
                "sha256": expected,
                "projection": list(PROJECTED_COLUMNS),
            },
            "representation": {
                "id": "v4_30_common_roi",
                "t0_t1_t2": True,
                "interval": "[start,end)",
                "event_pixel_diff": 5,
                "bins_per_polarity": 5,
                "shape": [3, 12, 128, 128],
            },
            "split": {
                "path": str(split_path.resolve()),
                "sha256": split_sha,
                "version": SPLIT_CONTRACT["version"],
            },
            "opened_paths": [str(source.resolve()), *(str(path) for path in sorted(readers))],
            "provenance": {"boxes_transient_only": True, "targets_opened": False},
        }
        (stage / "manifest.json").write_text(strict_json(manifest), encoding="utf-8")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/e_jepa_garl_object_event_operator_audit_v4_31.yaml",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--full", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--event-root", type=Path)
    a = p.parse_args()
    try:
        print(
            strict_json(
                build(
                    yaml.safe_load(a.config.read_text()),
                    a.output_dir,
                    full=a.full,
                    force=a.force,
                    event_root=a.event_root,
                )
            )
        )
    except Exception as exc:
        print(
            strict_json(
                {
                    "artifact_type": "object_event_v4_31_sanitized_cache_v1",
                    "status": "invalid_incomplete",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
