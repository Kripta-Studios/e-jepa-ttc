"""Resumable RAW16 count cache for the Stage 64 paired experiment."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact
from e_jepa_ttc.data.raw_event_binding import ReadOnlyTrainAccess, select_hash_probe_tokens

RAW_SHAPE_TAIL = (2, 16, 2, 64, 64)


@dataclass(frozen=True)
class RawCacheView:
    """Memory-mapped counts plus label-free observation metadata."""

    counts: np.ndarray
    durations_s: np.ndarray
    times: np.ndarray
    valid_patches: np.ndarray
    metadata: pd.DataFrame


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _file_sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_polarity(values: np.ndarray, encoding: str) -> np.ndarray:
    data = np.asarray(values)
    if encoding == "zero_one" and np.isin(data, (0, 1)).all():
        return 2 * data.astype(np.int8) - 1
    if encoding == "signed" and np.isin(data, (-1, 1)).all():
        return data.astype(np.int8)
    raise ValueError("raw polarity does not match its explicit file encoding")


def rasterize_bound_window(
    *,
    x: np.ndarray,
    y: np.ndarray,
    t_us: np.ndarray,
    polarity: np.ndarray,
    binding: pd.Series,
) -> tuple[np.ndarray, int]:
    """Hard-bin one exact raw window into ``[16,2,64,64]`` uint32 counts."""

    x_values = np.asarray(x, dtype=np.float64) + float(binding["event_x_offset_px"])
    y_values = np.asarray(y, dtype=np.float64)
    timestamps = np.asarray(t_us, dtype=np.int64)
    p_values = _signed_polarity(np.asarray(polarity), str(binding["polarity_encoding"]))
    if not (len(x_values) == len(y_values) == len(timestamps) == len(p_values)):
        raise ValueError("raw event arrays have inconsistent lengths")
    start = int(binding["window_start_us"])
    end = int(binding["window_end_us"])
    if np.any(timestamps < start) or np.any(timestamps >= end) or np.any(np.diff(timestamps) < 0):
        raise ValueError("raw slice violates its monotonic half-open time contract")
    x0, y0 = float(binding["roi_x0"]), float(binding["roi_y0"])
    x1, y1 = float(binding["roi_x1"]), float(binding["roi_y1"])
    gx = np.floor((x_values - x0) * 64.0 / (x1 - x0)).astype(np.int64)
    gy = np.floor((y_values - y0) * 64.0 / (y1 - y0)).astype(np.int64)
    keep = (gx >= 0) & (gx < 64) & (gy >= 0) & (gy < 64)
    if not np.any(keep):
        return np.zeros((16, 2, 64, 64), dtype=np.uint32), 0
    relative = timestamps[keep] - start
    bins = relative * 16 // (end - start)
    polarities = (p_values[keep] > 0).astype(np.int64)
    flat = ((bins * 2 + polarities) * 64 + gy[keep]) * 64 + gx[keep]
    counts = np.bincount(flat, minlength=16 * 2 * 64 * 64).reshape(16, 2, 64, 64)
    if counts.max(initial=0) > np.iinfo(np.uint32).max:
        raise OverflowError("one RAW16 cell exceeds uint32")
    return counts.astype(np.uint32), int(np.count_nonzero(keep))


def patch_support(counts: np.ndarray, minimum_events: int = 16) -> np.ndarray:
    """Return the common 4x4 support mask requiring events in both windows."""

    values = np.asarray(counts)
    if values.shape != RAW_SHAPE_TAIL or minimum_events <= 0:
        raise ValueError("expected counts [2,16,2,64,64]")
    spatial = values.sum(axis=(1, 2), dtype=np.uint64)
    patches = spatial.reshape(2, 4, 16, 4, 16).sum(axis=(2, 4), dtype=np.uint64)
    return (patches >= minimum_events).all(axis=0).reshape(16)


def _load_progress(path: Path, identity: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "identity": identity,
            "completed_tokens": 0,
            "max_cell_count": 0,
            "probe_records": {},
        }
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("identity") != identity:
        raise ValueError("raw-cache resume identity mismatch")
    progress.setdefault("probe_records", {})
    return progress


def build_raw_temporal_cache(
    *,
    binding: pd.DataFrame,
    raw_train_root: Path,
    access: ReadOnlyTrainAccess,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Build one shared uint16/uint32 memmap and its signed-by-hash index."""

    required = {
        "sample_token",
        "sequence_id",
        "track_id",
        "window_id",
        "start_event_index",
        "stop_event_index",
        "h5_file_sha256",
        "roi_transform_sha256",
    }
    if not required <= set(binding) or binding.duplicated(["sample_token", "window_id"]).any():
        raise ValueError("raw binding schema or uniqueness mismatch")
    metadata_source = cast(
        pd.DataFrame, binding[["sample_token", "sequence_id", "track_id", "outer_fold"]]
    )
    metadata = cast(
        pd.DataFrame,
        metadata_source.drop_duplicates().sort_values(by=["sample_token"]).reset_index(drop=True),
    )
    if len(binding) != 2 * len(metadata):
        raise ValueError("each token must have exactly two raw windows")
    ordered = binding.merge(metadata.reset_index(names="cache_row"), validate="many_to_one")
    ordered = ordered.sort_values(["cache_row", "window_id"]).reset_index(drop=True)
    identity_columns = [
        "sample_token",
        "window_id",
        "h5_file_sha256",
        "start_event_index",
        "stop_event_index",
        "roi_transform_sha256",
    ]
    identity = hashlib.sha256(
        ordered[identity_columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=resume)
    progress_path = output_dir / "build_progress.json"
    progress = _load_progress(progress_path, identity)
    temp_counts = output_dir / "raw_counts.uint32.tmp.npy"
    mode = "r+" if temp_counts.is_file() and resume else "w+"
    counts = np.lib.format.open_memmap(
        temp_counts, mode=mode, dtype=np.uint32, shape=(len(metadata), *RAW_SHAPE_TAIL)
    )
    durations = np.zeros((len(metadata), 2, 16), dtype=np.float64)
    times = np.zeros((len(metadata), 6), dtype=np.float32)
    valid = np.zeros((len(metadata), 16), dtype=np.bool_)
    roi_counts = np.zeros((len(metadata), 2), dtype=np.int64)
    # Reconstruct deterministic observation metadata on every resume; it is inexpensive.
    grouped = {
        int(cast(Any, key)): cast(pd.DataFrame, value)
        for key, value in ordered.groupby("cache_row", sort=True)
    }
    probe_tokens = set(select_hash_probe_tokens(binding, min(64, len(metadata))))
    began = time.perf_counter()
    completed = int(progress["completed_tokens"])
    handles: dict[str, h5py.File] = {}
    try:
        for row_index in range(len(metadata)):
            rows = grouped[row_index].sort_values("window_id")
            durations_us = np.asarray(
                rows["window_end_us"].to_numpy(np.int64)
                - rows["window_start_us"].to_numpy(np.int64)
            )
            edges = [
                np.asarray(
                    [int(start) + int(duration) * k // 16 for k in range(17)], dtype=np.int64
                )
                for start, duration in zip(rows["window_start_us"], durations_us, strict=True)
            ]
            for window in range(2):
                durations[row_index, window] = np.diff(edges[window]) * 1e-6
            r0, r1 = (rows.iloc[0], rows.iloc[1])
            times[row_index] = np.asarray(
                [
                    durations_us[0] * 1e-6,
                    durations_us[1] * 1e-6,
                    (int(r0["window_end_us"]) - int(r1["window_end_us"])) * 1e-6,
                    (int(r1["window_start_us"]) - int(r0["window_end_us"])) * 1e-6,
                    int(r0["endpoint_delta01_us"]) * 1e-6,
                    int(r0["endpoint_delta12_us"]) * 1e-6,
                ],
                dtype=np.float32,
            ) / np.float32(0.1)
            if row_index < completed:
                valid[row_index] = patch_support(np.asarray(counts[row_index]))
                roi_counts[row_index] = np.asarray(counts[row_index]).sum(
                    axis=(1, 2, 3, 4), dtype=np.uint64
                )
                continue
            token_counts = np.zeros(RAW_SHAPE_TAIL, dtype=np.uint32)
            token = str(rows.iloc[0]["sample_token"])
            token_probe = hashlib.sha256()
            token_probe_events = 0
            for window, (_, item) in enumerate(rows.iterrows()):
                sequence = str(item["sequence_id"])
                handle = handles.get(sequence)
                if handle is None:
                    path = access.event_path(sequence, f"{sequence}/events.h5", "cache_h5_open")
                    handle = h5py.File(path, "r")
                    handles[sequence] = handle
                start_idx, stop_idx = int(item["start_event_index"]), int(item["stop_event_index"])
                path = raw_train_root / sequence / "events.h5"
                access.record_h5_slice(path, "cache_raw_slice", start_idx, stop_idx)
                group = cast(h5py.Group, handle["events"])
                arrays = [
                    np.asarray(cast(h5py.Dataset, group[key])[start_idx:stop_idx]) for key in "xytp"
                ]
                value, roi_count = rasterize_bound_window(
                    x=arrays[0], y=arrays[1], t_us=arrays[2], polarity=arrays[3], binding=item
                )
                token_counts[window] = value
                roi_counts[row_index, window] = roi_count
                if token in probe_tokens:
                    for array in arrays:
                        if len(array):
                            token_probe.update(array[:1].tobytes())
                            token_probe.update(array[-1:].tobytes())
                    token_probe_events += len(arrays[2])
            counts[row_index] = token_counts
            valid[row_index] = patch_support(token_counts)
            if token in probe_tokens:
                progress["probe_records"][token] = {
                    "events": token_probe_events,
                    "endpoint_checksum": token_probe.hexdigest(),
                }
            progress["completed_tokens"] = row_index + 1
            progress["max_cell_count"] = max(
                int(progress["max_cell_count"]), int(token_counts.max(initial=0))
            )
            if (row_index + 1) % 16 == 0 or row_index + 1 == len(metadata):
                counts.flush()
                _atomic_json(progress_path, progress)
    finally:
        for handle in handles.values():
            handle.close()
        counts.flush()
        del counts
    max_cell = int(progress["max_cell_count"])
    dtype: Literal["uint16", "uint32"] = "uint16" if max_cell <= 65535 else "uint32"
    final_path = output_dir / f"raw_counts.{dtype}.npy"
    if not final_path.is_file():
        source = np.load(temp_counts, mmap_mode="r")
        target = np.lib.format.open_memmap(
            final_path.with_suffix(final_path.suffix + ".tmp"),
            mode="w+",
            dtype=np.dtype(dtype),
            shape=source.shape,
        )
        for start in range(0, len(source), 16):
            target[start : start + 16] = source[start : start + 16]
        target.flush()
        del target, source
        os.replace(final_path.with_suffix(final_path.suffix + ".tmp"), final_path)
    np.save(output_dir / "durations_s.npy", durations)
    np.save(output_dir / "times_scaled_0p1s.npy", times)
    np.save(output_dir / "valid_patches.npy", valid)
    np.save(output_dir / "roi_event_counts.npy", roi_counts)
    metadata.to_csv(output_dir / "metadata.csv", index=False, lineterminator="\n")
    elapsed = time.perf_counter() - began
    probe_records = progress["probe_records"]
    if set(probe_records) != probe_tokens:
        raise RuntimeError("raw physical-read probe is incomplete after cache construction")
    probe_payload = json.dumps(probe_records, sort_keys=True, separators=(",", ":"))
    manifest = {
        "artifact_type": "scientific_recovery_v9_raw16_temporal_cache_v1",
        "status": "completed",
        "identity_sha256": identity,
        "shape": [len(metadata), *RAW_SHAPE_TAIL],
        "dtype": dtype,
        "counts": {
            "path": final_path.name,
            "bytes": final_path.stat().st_size,
            "sha256": _file_sha256(final_path),
            "max_cell_count": max_cell,
            "overflow": max_cell > 65535,
        },
        "tokens": len(metadata),
        "windows": 2 * len(metadata),
        "supported_tokens": int(valid.any(axis=1).sum()),
        "supported_fraction": float(valid.any(axis=1).mean()),
        "empty_roi_windows": int((roi_counts == 0).sum()),
        "roi_events": int(roi_counts.sum()),
        "read_probe": {
            "selection": "sha256_ranked_sequence_stratified",
            "tokens": len(probe_tokens),
            "events": int(sum(record["events"] for record in probe_records.values())),
            "checksum": hashlib.sha256(probe_payload.encode("utf-8")).hexdigest(),
        },
        "build_seconds_this_invocation": elapsed,
        "contains_targets": False,
        "common_cache_for_all_arms": True,
    }
    manifest = sign_stage63_65_artifact(manifest, evidence_type="physical_raw_cache")
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def load_raw_cache(path: Path) -> RawCacheView:
    """Load and validate a completed raw cache without materializing the count tensor."""

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not verify_artifact_hash(manifest) or manifest.get("status") != "completed":
        raise ValueError("raw cache is not complete")
    count_path = path / str(manifest["counts"]["path"])
    if _file_sha256(count_path) != manifest["counts"]["sha256"]:
        raise ValueError("raw cache count hash mismatch")
    counts = np.load(count_path, mmap_mode="r")
    durations = np.load(path / "durations_s.npy")
    times = np.load(path / "times_scaled_0p1s.npy")
    valid = np.load(path / "valid_patches.npy")
    metadata = pd.read_csv(path / "metadata.csv", dtype={"sample_token": str})
    n = len(metadata)
    if counts.shape != (n, *RAW_SHAPE_TAIL) or durations.shape != (n, 2, 16):
        raise ValueError("raw cache arrays are row-misaligned")
    if times.shape != (n, 6) or valid.shape != (n, 16):
        raise ValueError("raw cache observation metadata is row-misaligned")
    return RawCacheView(counts, durations, times, valid, metadata)


def fold_rate_normalization(
    cache: RawCacheView, train_rows: np.ndarray, *, chunk_size: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute outer-train-only mean/std of log1p count rates per polarity."""

    rows = np.asarray(train_rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0 or np.any(rows < 0) or np.any(rows >= len(cache.metadata)):
        raise ValueError("invalid normalization row indices")
    total = np.zeros(2, dtype=np.float64)
    square = np.zeros(2, dtype=np.float64)
    count = 0
    for start in range(0, len(rows), chunk_size):
        index = rows[start : start + chunk_size]
        raw = np.asarray(cache.counts[index], dtype=np.float64)
        duration = cache.durations_s[index, :, :, None, None, None]
        value = np.log1p(raw / duration)
        total += value.sum(axis=(0, 1, 2, 4, 5))
        square += np.square(value).sum(axis=(0, 1, 2, 4, 5))
        count += value.shape[0] * value.shape[1] * value.shape[2] * value.shape[4] * value.shape[5]
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-8):
        raise ValueError("invalid outer-train rate normalization")
    return mean.astype(np.float32), std.astype(np.float32)


__all__ = [
    "RAW_SHAPE_TAIL",
    "RawCacheView",
    "build_raw_temporal_cache",
    "fold_rate_normalization",
    "load_raw_cache",
    "patch_support",
    "rasterize_bound_window",
]
