"""Exact right-RGB overlap: clocks propose candidates, all pixels prove identity.

This narrowly scoped audit reads the existing reader's HDF5 RGB paths only.
It does not parse events, calibration, labels, or create a second sensor adapter.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _rgb(handle: h5py.File) -> tuple[np.ndarray, h5py.Dataset]:
    times = np.asarray(handle["blackflys/right/ts"])
    frames = handle["blackflys/right/data"]
    if (
        times.ndim != 1
        or times.dtype.kind not in "iu"
        or len(times) == 0
        or np.any(np.diff(times.astype(np.int64)) <= 0)
        or not isinstance(frames, h5py.Dataset)
        or len(frames) != len(times)
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
    ):
        raise ValueError("RGB requires increasing integer clocks and aligned uint8 NHWC frames")
    return times.astype(np.int64), frames


def match_recording(competition_hdf5: Path, source_hdf5: Path) -> dict[str, Any]:
    """Compare every pixel in every clock-compatible contiguous segment, bounded by frame."""
    matches: list[dict[str, Any]] = []
    clock_candidates = 0
    with h5py.File(competition_hdf5, "r") as crop, h5py.File(source_hdf5, "r") as source:
        ct, cf = _rgb(crop)
        st, sf = _rgb(source)
        count = len(ct)
        if sf.shape[1:] == cf.shape[1:]:
            relative = ct - ct[0]
            for start in range(len(st) - count + 1):
                if not np.array_equal(st[start : start + count] - st[start], relative):
                    continue
                clock_candidates += 1
                frames = []
                for index in range(count):
                    left, right = cf[index], sf[start + index]
                    if not np.array_equal(left, right):
                        break
                    frames.append({
                        "competition_frame": index,
                        "original_frame": start + index,
                        "equal": True,
                        "competition_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
                        "original_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
                    })
                else:
                    matches.append({
                        "original_start_frame": start,
                        "timestamp_offset_us": int(st[start]) - int(ct[0]),
                        "all_query_rgb_timestamps_match": True,
                        "all_rgb_frames": frames,
                    })
    return {
        "status": "unique" if len(matches) == 1 else "ambiguous" if matches else "no_match",
        "query_frames": count,
        "exact_clock_candidates": clock_candidates,
        "matches": matches,
    }


def match_sources(competition_hdf5: Path, sources: Mapping[str, Path]) -> dict[str, Any]:
    """Require global uniqueness across all supplied recordings, not just one file."""
    matches = []
    candidates = 0
    count = 0
    for sequence_id, source in sorted(sources.items()):
        result = match_recording(competition_hdf5, source)
        candidates += result["exact_clock_candidates"]
        count = result["query_frames"]
        matches.extend({"sequence_id": sequence_id, **item} for item in result["matches"])
    return {
        "status": "unique" if len(matches) == 1 else "ambiguous" if matches else "no_match",
        "query_frames": count,
        "exact_clock_candidates": candidates,
        "matches": matches,
    }


def compare_reference(result: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    """Compare scientific fields, excluding timestamps, machine paths and sample summaries."""
    actual = result["matches"]
    expected = reference["verified_sample_matches"]
    fields = ("sequence_id", "original_start_frame", "timestamp_offset_us", "all_rgb_frames")
    return result["status"] == "unique" and len(expected) == 1 and all(
        actual[0][field] == expected[0][field] for field in fields
    )
