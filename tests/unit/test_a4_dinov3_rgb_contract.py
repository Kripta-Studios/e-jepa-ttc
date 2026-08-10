"""Unit tests for the A4 RGB↔event endpoint binding contract."""

from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.data.event_v4_geometry import box_in_common_roi
from scripts.materialize_dinov3_relational_teacher import _resolve_rgb_endpoints


def _record_and_row() -> tuple[dict[str, object], dict[str, object]]:
    boxes = [
        (10.0, 10.0, 20.0, 20.0),
        (20.0, 20.0, 40.0, 40.0),  # t1 selected by event cache
        (30.0, 30.0, 50.0, 50.0),
        (35.0, 35.0, 60.0, 60.0),
        (40.0, 40.0, 80.0, 80.0),  # t2 selected by event cache
        (45.0, 45.0, 90.0, 90.0),
    ]
    square = (0.0, 0.0, 100.0, 100.0)
    mapped_t1 = box_in_common_roi(boxes[1], square, roi_size=128)
    mapped_t2 = box_in_common_roi(boxes[4], square, roi_size=128)
    record: dict[str, object] = {
        "event_v4_common_square_xyxy": np.asarray(square, dtype=np.float32),
        "event_v4_boxes_xyxy": np.asarray(
            [[0.0, 0.0, 1.0, 1.0], mapped_t1, mapped_t2], dtype=np.float32
        ),
        "event_v4_common_roi": np.zeros((3, 12, 128, 128), dtype=np.float32),
        "garl_delta_t_s": 0.3,
    }
    timestamps = [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000]
    row: dict[str, object] = {
        "rgb_shard_paths": [f"s{i}.tar" for i in range(6)],
        "rgb_member_paths": [f"f{i}.jpg" for i in range(6)],
        "frame_timestamps_us": timestamps,
        "event_windows_us": [[t - 50_000, t] for t in timestamps],
        "boxes_xyxy": boxes,
    }
    return record, row


def test_resolve_rgb_endpoints_binds_exact_event_cache_pair() -> None:
    record, row = _record_and_row()
    resolved = _resolve_rgb_endpoints(record, row)
    assert resolved["indices"] == (1, 4)
    assert resolved["rgb_shards"] == ("s1.tar", "s4.tar")
    assert resolved["rgb_members"] == ("f1.jpg", "f4.jpg")
    assert resolved["frame_timestamps_us"] == (1_100_000, 1_400_000)


def test_resolve_rgb_endpoints_rejects_unaligned_metadata() -> None:
    record, row = _record_and_row()
    row["rgb_member_paths"] = list(row["rgb_member_paths"])[:-1]  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unaligned RGB/event metadata lengths"):
        _resolve_rgb_endpoints(record, row)


def test_resolve_rgb_endpoints_rejects_delta_t_mismatch() -> None:
    record, row = _record_and_row()
    record["garl_delta_t_s"] = 0.2
    with pytest.raises(ValueError, match="could not uniquely bind RGB endpoints"):
        _resolve_rgb_endpoints(record, row)
