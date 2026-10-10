"""Label-free native TRAIN40 geometry applied to official GarlTTC test inputs."""

# ruff: noqa: ANN401 -- public parquet rows contain heterogeneous nested metadata.
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes, shifted_precontext_window
from e_jepa_ttc.data.garlttc_eap import normalize_boxes_xyxy, normalize_event_windows_us
from e_jepa_ttc.data.garlttc_lhr_cache import select_temporal_indices

LAGS_US = np.arange(350_000, -1, -50_000, dtype=np.int64)
INPUT_COLUMNS = (
    "sequence_id",
    "sample_token",
    "timestamp_us",
    "frame_timestamps_us",
    "events_path",
    "event_windows_us",
    "rgb_member_paths",
    "boxes_xyxy",
)
EXPOSURE_COLUMNS = (
    "sequence_id",
    "rgb_member_path",
    "rgb_exposure_start_timestamp_us",
    "rgb_exposure_end_timestamp_us",
)


def describe_input(
    row: Mapping[str, Any],
    exposures: Mapping[tuple[str, str], Mapping[str, Any]],
    raw_root: Path,
    *,
    split: str,
) -> dict[str, Any]:
    """Preserve endpoint selection, sensor clock, common crop and ROI availability.

    Only public sensor/ROI columns are consumed. Historical windows reuse the
    supplied current crop, exactly as in TRAIN40; they are not historical boxes.
    """
    if split not in {"train", "test"}:
        raise ValueError("Expected train or test split")
    sequence = str(row["sequence_id"])
    relative = str(row["events_path"]).replace("\\", "/")
    if relative.split("/") != ["data", split, sequence, "events.h5"]:
        raise ValueError("Event path must identify exactly the supplied split and sequence")
    raw = (raw_root / relative).resolve()
    if raw_root.resolve() not in raw.parents:
        raise ValueError("Event path escapes media root")
    boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
    windows = normalize_event_windows_us(row["event_windows_us"])
    stamps = [int(x) for x in row["frame_timestamps_us"]]
    members = [str(x) for x in row["rgb_member_paths"]]
    if len({len(boxes), len(windows), len(stamps), len(members)}) != 1:
        raise ValueError("Unaligned public temporal metadata")
    first, second, context = select_temporal_indices(
        stamps,
        anchor_timestamp_us=int(row["timestamp_us"]),
        target_delta_t_s=0.1,
        tolerance_s=0.025,
        context_delta_t_s=0.1,
        context_tolerance_s=0.05,
    )
    selected = (first, second)
    frames = [exposures[(sequence, members[i])] for i in selected]
    if [int(f["rgb_exposure_start_timestamp_us"]) for f in frames] != [
        windows[i][1] for i in selected
    ]:
        raise ValueError("Exposure metadata and raw event clock disagree")
    available = max(int(f["rgb_exposure_end_timestamp_us"]) for f in frames)
    base = np.asarray(
        [
            windows[context]
            if context is not None
            else shifted_precontext_window(
                windows[first],
                shift_s=0.1,
            ),
            windows[first],
            windows[second],
        ],
        dtype=np.int64,
    )
    anchor = int(base[-1, 1])
    if available < anchor or np.any(base[:, 1] <= base[:, 0]):
        raise ValueError("Invalid source windows or exposure availability")
    if np.any(np.diff(base[:, 1]) <= 0) or int(base.max()) != anchor:
        raise ValueError("Noncausal source windows")
    return {
        "sample_token": str(row["sample_token"]),
        "sequence": sequence,
        "path": str(raw),
        "windows": base.tolist(),
        "square": list(common_square_from_boxes(boxes, selected, margin_fraction=0.25)),
        "delta": (stamps[second] - stamps[first]) * 1e-6,
        "anchor": anchor,
        "available": available,
        "precontext": "annotated" if context is not None else "shifted_real_events",
    }


def supported_job(job: Mapping[str, Any], source_start: int, source_end: int) -> dict[str, Any]:
    """Mask only incomplete leading history, retaining every supported event."""
    windows = np.asarray(job["windows"], dtype=np.int64)
    valid = windows.min() - LAGS_US >= source_start
    if not valid[-1] or windows.max() > source_end:
        raise ValueError("Current query lacks complete raw event support")
    return {
        **job,
        "windows": windows,
        "square": np.asarray(job["square"], np.float64),
        "valid": valid,
    }
