"""Materialize compact ML tensors from indexed event windows."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile

from e_jepa_ttc.data.evttc import (
    NAVIGATION_FEATURE_NAMES,
    read_events_window,
    read_manifest,
    read_navigation_window_features,
)
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid
from e_jepa_ttc.utils.io import ensure_parent, read_structured, write_structured


def validate_voxel_cache(cache: NpzFile) -> None:
    if "cache_format_version" not in cache.files or int(cache["cache_format_version"]) < 2:
        msg = "Training requires cache_format_version >= 2."
        raise ValueError(msg)
    for key in [
        "normalization",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "preprocessing_config_sha256",
    ]:
        if key not in cache.files:
            msg = f"Cache missing required metadata field: {key}"
            raise ValueError(msg)

    # Sparse occupancy audit (Requirement 4.4)
    x = cache["x"]
    if x.size > 0:
        # Check a small random sample (up to 100 windows) to ensure it's not completely empty.
        # Use a fixed seed for deterministic audit behaviour.
        rng = np.random.default_rng(42)
        idx = rng.choice(x.shape[0], size=min(x.shape[0], 100), replace=False)
        sample = x[idx]
        if np.count_nonzero(sample) == 0:
            msg = (
                "Sparse occupancy audit failed: "
                "Sampled cache tensors are completely zero (no events)."
            )
            raise ValueError(msg)


def _load_windows(index_path: str | Path) -> list[dict[str, Any]]:
    data = read_structured(index_path)
    windows = data.get("windows")
    if not isinstance(windows, list):
        msg = f"Index {index_path} does not contain a windows list."
        raise ValueError(msg)
    return [dict(item) for item in windows]


def _downsample_events(events: EventBatch, *, width: int, height: int) -> EventBatch:
    if events.width == width and events.height == height:
        return events
    if events.num_events == 0:
        return EventBatch.empty(
            width=width,
            height=height,
            sequence_id=events.sequence_id,
            t_start_us=events.t_start_us,
            t_end_us=events.t_end_us,
        )
    x = np.floor(events.x.astype(np.float64) * width / events.width).astype(np.int32)
    y = np.floor(events.y.astype(np.float64) * height / events.height).astype(np.int32)
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    return EventBatch(
        x=x,
        y=y,
        t_us=events.t_us,
        polarity=events.polarity,
        width=width,
        height=height,
        sequence_id=events.sequence_id,
        t_start_us=events.t_start_us,
        t_end_us=events.t_end_us,
    )


def _metadata_channels(events: EventBatch, *, width: int, height: int) -> np.ndarray:
    duration_s = max(events.duration_us / 1_000_000.0, 1e-6)
    log_count = np.float32(np.log1p(events.num_events))
    log_rate = np.float32(np.log1p(events.num_events / duration_s))
    meta = np.empty((2, height, width), dtype=np.float32)
    meta[0].fill(log_count)
    meta[1].fill(log_rate)
    return meta


def _constant_channels(values: np.ndarray, *, width: int, height: int) -> np.ndarray:
    channels = np.empty((int(values.shape[0]), height, width), dtype=np.float32)
    for idx, value in enumerate(values.astype(np.float32, copy=False)):
        channels[idx].fill(value)
    return channels


def _hash_file(filepath: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_voxel_cache(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    width: int = 160,
    height: int = 90,
    bins: int = 5,
    normalize: bool = True,
    metadata_channels: bool = False,
    navigation_channels: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build an `.npz` cache with voxel tensors and labels."""

    if width <= 0 or height <= 0 or bins <= 0:
        msg = "width, height and bins must be positive."
        raise ValueError(msg)

    sequences = {sequence.sequence_id: sequence for sequence in read_manifest(manifest_path)}
    split_data = read_structured(split_path).get("splits", {})
    if not isinstance(split_data, dict):
        msg = f"Split file {split_path} does not contain splits."
        raise ValueError(msg)
    split_for_sequence = {
        str(sequence_id): str(split_name)
        for split_name, sequence_ids in split_data.items()
        for sequence_id in sequence_ids
    }

    windows = _load_windows(index_path)
    if limit is not None:
        windows = windows[:limit]
    channels = (
        bins * 2
        + (2 if metadata_channels else 0)
        + (len(NAVIGATION_FEATURE_NAMES) if navigation_channels else 0)
    )
    x_out = np.empty((len(windows), channels, height, width), dtype=np.float16)
    y_ttc = np.empty((len(windows),), dtype=np.float32)
    timestamps = np.empty((len(windows),), dtype=np.int64)
    context_start_us = np.empty((len(windows),), dtype=np.int64)
    context_end_us = np.empty((len(windows),), dtype=np.int64)
    sequence_ids: list[str] = []
    splits: list[str] = []
    event_counts = np.empty((len(windows),), dtype=np.int32)

    start_time = time.perf_counter()
    for idx, window in enumerate(windows):
        sequence_id = str(window["sequence_id"])
        sequence = sequences[sequence_id]
        event_hdf5 = sequence.resolve("event_hdf5")
        if event_hdf5 is None:
            msg = f"Sequence {sequence_id} has no event_hdf5."
            raise ValueError(msg)
        events = read_events_window(
            event_hdf5,
            t_start_us=int(window["context_start_us"]),
            t_end_us=int(window["context_end_us"]),
            sequence_id=sequence_id,
        )
        small = _downsample_events(events, width=width, height=height)
        voxel = encode_voxel_grid(small, bins=bins, normalize=normalize)
        if metadata_channels:
            voxel = np.concatenate([voxel, _metadata_channels(events, width=width, height=height)])
        if navigation_channels:
            navigation = read_navigation_window_features(
                event_hdf5,
                t_start_us=int(window["context_start_us"]),
                t_end_us=int(window["context_end_us"]),
            )
            voxel = np.concatenate(
                [voxel, _constant_channels(navigation, width=width, height=height)]
            )
        x_out[idx] = voxel.astype(np.float16)
        y_ttc[idx] = float(window["ttc_seconds"])
        timestamps[idx] = int(window["timestamp_us"])
        context_start_us[idx] = int(window["context_start_us"])
        context_end_us[idx] = int(window["context_end_us"])
        sequence_ids.append(sequence_id)
        splits.append(split_for_sequence.get(sequence_id, "unassigned"))
        event_counts[idx] = events.num_events

    output = ensure_parent(output_path)
    np.savez(
        output,
        x=x_out,
        y_ttc=y_ttc,
        timestamp_us=timestamps,
        context_start_us=context_start_us,
        context_end_us=context_end_us,
        sequence_id=np.array(sequence_ids),
        split=np.array(splits),
        event_count=event_counts,
        width=np.array(width, dtype=np.int32),
        height=np.array(height, dtype=np.int32),
        bins=np.array(bins, dtype=np.int32),
        normalize=np.array(normalize, dtype=np.bool_),
        metadata_channels=np.array(metadata_channels, dtype=np.bool_),
        navigation_channels=np.array(navigation_channels, dtype=np.bool_),
        navigation_feature_names=np.array(NAVIGATION_FEATURE_NAMES),
        future_window_semantics=np.array("disjoint_window_start_after_context_plus_horizon"),
        cache_format_version=np.array(2, dtype=np.int64),
        normalization=np.array("non_centered_occupied_p95_scale" if normalize else "none"),
        source_manifest_sha256=np.array(_hash_file(manifest_path)),
        split_manifest_sha256=np.array(_hash_file(split_path)),
        preprocessing_config_sha256=np.array(_hash_file(__file__)),
    )
    summary = {
        "output": output.as_posix(),
        "window_count": int(len(windows)),
        "shape": list(x_out.shape),
        "dtype": str(x_out.dtype),
        "width": width,
        "height": height,
        "bins": bins,
        "normalize": normalize,
        "metadata_channels": metadata_channels,
        "navigation_channels": navigation_channels,
        "navigation_feature_names": list(NAVIGATION_FEATURE_NAMES),
        "future_window_semantics": "disjoint_window_start_after_context_plus_horizon",
        "seconds": time.perf_counter() - start_time,
        "mean_events_per_window": float(np.mean(event_counts)) if len(event_counts) else 0.0,
    }
    write_structured(output.with_suffix(".summary.json"), summary)
    return summary


def remap_cache_splits(
    *,
    cache_path: str | Path,
    split_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Copy an NPZ cache while replacing split labels by sequence id."""

    cache = np.load(cache_path, allow_pickle=False)
    if "sequence_id" not in cache.files or "split" not in cache.files:
        msg = "Cache must contain sequence_id and split arrays."
        raise ValueError(msg)
    split_data = read_structured(split_path).get("splits", {})
    if not isinstance(split_data, dict):
        msg = f"Split file {split_path} does not contain splits."
        raise ValueError(msg)
    split_for_sequence = {
        str(sequence_id): str(split_name)
        for split_name, sequence_ids in split_data.items()
        for sequence_id in sequence_ids
    }
    sequence_ids = cache["sequence_id"].astype(str)
    old_split = cache["split"].astype(str)
    new_split = np.array(
        [split_for_sequence.get(str(sequence_id), "unassigned") for sequence_id in sequence_ids],
        dtype=str,
    )
    arrays = {key: cache[key] for key in cache.files if key != "split"}
    output = ensure_parent(output_path)
    np.savez(output, **arrays, split=new_split)
    old_counts = {
        str(name): int(np.sum(old_split == name)) for name in sorted(set(old_split.tolist()))
    }
    new_counts = {
        str(name): int(np.sum(new_split == name)) for name in sorted(set(new_split.tolist()))
    }
    summary = {
        "input": Path(cache_path).as_posix(),
        "output": output.as_posix(),
        "split": Path(split_path).as_posix(),
        "window_count": int(new_split.shape[0]),
        "old_split_counts": old_counts,
        "new_split_counts": new_counts,
        "unassigned_count": int(np.sum(new_split == "unassigned")),
    }
    write_structured(output.with_suffix(".summary.json"), summary)
    return summary
