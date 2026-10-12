"""Label-free native TRAIN40 event observations and same-track context selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes, shifted_precontext_window
from e_jepa_ttc.data.garlttc_eap import normalize_boxes_xyxy, normalize_event_windows_us
from e_jepa_ttc.data.garlttc_lhr_cache import select_temporal_indices
from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader
from operational.rgb_port.infer_experts import bounded_event_history
from operational.train40_system.prepare import configuration

EVENT_COLUMNS = (
    "sequence_id",
    "sample_token",
    "track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "event_windows_us",
    "events_path",
    "boxes_xyxy",
)


def native_job(row: dict, root: Path) -> dict:
    """Select exactly the native producer windows and ROI, without a target field."""
    config = configuration()
    boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
    windows = normalize_event_windows_us(row["event_windows_us"])
    stamps = [int(x) for x in row["frame_timestamps_us"]]
    usable = min(len(boxes), len(windows), len(stamps))
    assert usable >= 2
    boxes, windows, stamps = boxes[:usable], windows[:usable], stamps[:usable]
    first, second, context = select_temporal_indices(
        stamps,
        anchor_timestamp_us=int(row["timestamp_us"]),
        target_delta_t_s=config.target_delta_t_s,
        tolerance_s=config.delta_t_tolerance_s,
        context_delta_t_s=config.jepa_context_delta_t_s,
        context_tolerance_s=config.jepa_context_tolerance_s,
    )
    precontext = (
        windows[context]
        if context is not None
        else shifted_precontext_window(windows[first], shift_s=config.jepa_context_delta_t_s)
    )
    selected = np.asarray([precontext, windows[first], windows[second]], dtype=np.int64)
    span = int(selected[-1, 1] - selected[0, 0])
    assert 0 < span <= 650000 and np.all(selected[:, 1] > selected[:, 0])
    path = (root / str(row["events_path"])).resolve()
    assert path.is_relative_to(root.resolve())
    return dict(
        sample_token=str(row["sample_token"]),
        sequence_id=str(row["sequence_id"]),
        track_id=str(row["track_id"]),
        anchor_us=int(row["timestamp_us"]),
        available_us=int(row["timestamp_us"]),
        input_span_us=span,
        sensor_anchor_us=int(selected[-1, 1]),
        windows_us=selected.tolist(),
        square_xyxy=list(
            common_square_from_boxes(
                boxes, (first, second), margin_fraction=config.event_v4_margin_fraction
            )
        ),
        delta_t_s=(stamps[second] - stamps[first]) * 1e-6,
        raw_path=str(path),
        precontext="annotated" if context is not None else "shifted_real_events",
    )


def native_histories(jobs: list[dict]) -> dict[str, list[str]]:
    """Retain actual same-object observations, including each producer's full input span."""
    observations = {r["sample_token"]: r for r in jobs}
    assert len(observations) == len(jobs)
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for row in jobs:
        groups.setdefault((row["sequence_id"], row["track_id"]), []).append(
            (row["anchor_us"], row["sample_token"])
        )
    for timeline in groups.values():
        timeline.sort()
        assert len({r[0] for r in timeline}) == len(timeline), "Ambiguous same-object timestamp"
    result = {}
    for row in jobs:
        timeline = groups[row["sequence_id"], row["track_id"]]
        result[row["sample_token"]] = bounded_event_history(
            timeline, [r[0] for r in timeline], observations, row["anchor_us"]
        )
        assert result[row["sample_token"]][-1] == row["sample_token"]
    return result


def native_event(job: dict, reader: CachedEventReader) -> torch.Tensor:
    """Use unchanged ROI projection and canonical voxelization on each true sensor window."""
    config = configuration()
    frames = []
    for start, end in job["windows_us"]:
        raw = reader.read_window(start, end)
        mapped = map_roi(raw, tuple(job["square_xyxy"]), config.roi_size, config.event_pixel_diff)
        frames.append(encode_mapped(mapped, start, end, config.roi_size, job["sequence_id"]))
    result = torch.stack(frames)
    assert result.shape == (3, 12, 128, 128) and torch.isfinite(result).all()
    return result
