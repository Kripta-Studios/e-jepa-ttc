"""Exact H8 mapped-union encoding with recursively bounded raw retention."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.simplex_t.context_raw_union import _read_crop


@dataclass(frozen=True)
class _Task:
    start: int
    end: int
    slot: int
    window: int


def _tasks(windows: np.ndarray, lags: np.ndarray, valid: np.ndarray) -> list[_Task]:
    if windows.shape != (3, 2) or lags.shape != (16,) or valid.shape != (16,):
        raise ValueError("registered H16 window layout required")
    if valid.dtype != np.bool_ or valid[:8].any() or not valid[-1]:
        raise ValueError("H8 requires eight reserved leading slots and a present current slot")
    present = np.flatnonzero(valid)
    if len(present) > 8 or (np.diff(present) != 1).any():
        raise ValueError("H8 valid slots must be a contiguous suffix of the final eight slots")
    if windows.dtype.kind not in "iu" or lags.dtype.kind not in "iu":
        raise ValueError("H8 windows and lags must use integer microseconds")
    intervals = windows[None] - lags[:, None, None]
    tasks = [
        _Task(int(start), int(end), int(slot), int(window))
        for slot in present
        for window, (start, end) in enumerate(intervals[slot])
    ]
    if any(task.end <= task.start for task in tasks):
        raise ValueError("H8 requires positive half-open windows")
    return sorted(tasks, key=lambda task: (task.start, task.end, task.slot, task.window))


def encode_union(
    reader: Any,  # noqa: ANN401 - EAP reader protocol is supplied by the frozen loader
    windows: np.ndarray,
    lags: np.ndarray,
    valid: np.ndarray,
    roi: tuple,
    *,
    sequence_id: str,
    roi_size: int,
    event_pixel_diff: float,
    retained_bytes_max: int = 256 * 1024**2,
    profile: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Encode H8 exactly while recursively splitting only oversized raw unions."""
    tasks = _tasks(windows, lags, valid)
    if retained_bytes_max < 1 or roi_size < 1:
        raise ValueError("positive retained-byte cap and ROI size required")
    result = torch.zeros(16, 3, 12, roi_size, roi_size)
    metrics: dict[str, Any] = {
        "schema": "train40_h8_bounded_union_profile_v1",
        "retained_bytes_max": int(retained_bytes_max),
        "task_count": len(tasks),
        "read_attempts": 0,
        "successful_groups": 0,
        "split_count": 0,
        "peak_group_retained_bytes": 0,
        "groups": [],
    }
    started = time.perf_counter()

    def encode_group(group: list[_Task]) -> None:
        first = min(task.start for task in group)
        last = max(task.end for task in group)
        metrics["read_attempts"] += 1
        raw = _read_crop(
            reader,
            first,
            last,
            roi,
            event_pixel_diff,
            retained_bytes_max,
        )
        if raw is None:
            if len(group) == 1:
                raise RuntimeError("RESOURCE_PAUSE:single window retained byte cap")
            metrics["split_count"] += 1
            middle = len(group) // 2
            encode_group(group[:middle])
            encode_group(group[middle:])
            return

        retained = sum(value.nbytes for value in raw.values())
        metrics["peak_group_retained_bytes"] = max(
            int(metrics["peak_group_retained_bytes"]), retained
        )
        mapped = map_roi(raw, roi, roi_size, event_pixel_diff)
        del raw
        try:
            for task in group:
                result[task.slot, task.window] = encode_mapped(
                    mapped,
                    task.start,
                    task.end,
                    roi_size,
                    sequence_id,
                )
            metrics["successful_groups"] += 1
            metrics["groups"].append(
                {
                    "start_us": first,
                    "end_us": last,
                    "task_count": len(group),
                    "retained_bytes": retained,
                }
            )
        finally:
            del mapped

    encode_group(tasks)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    if profile is not None:
        profile.clear()
        profile.update(metrics)
    return result


def prepare(job: dict[str, Any]) -> np.ndarray:
    """Standalone worker callback using the canonical initialized H8 reader pool."""
    from operational.train40_system import history_resources8

    pool = history_resources8._reader_pool
    if pool is None:
        raise RuntimeError("H8 raw worker is not initialized")
    tensor = encode_union(
        pool.get(job["path"]),
        job["windows"],
        np.arange(15, -1, -1, dtype=np.int64) * 50_000,
        np.concatenate((np.zeros(8, dtype=bool), job["valid"])),
        tuple(job["square"]),
        sequence_id=job["sequence"],
        roi_size=128,
        event_pixel_diff=5,
        retained_bytes_max=int(job.get("retained_bytes_max", 256 * 1024**2)),
        profile=job.get("profile"),
    )
    return tensor.numpy()


__all__ = ["encode_union", "prepare"]
