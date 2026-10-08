from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from operational.evttc_rgb_transfer.inputs import CONTRACT, _nearest_frame_indices, prepare
from operational.evttc_transfer.inputs import _canonical_sha256, _sha256_file


def _fixture(tmp_path: Path, *, sync_offsets_us: tuple[int, int]) -> dict:
    raw_path = tmp_path / "events.hdf5"
    event_t = np.arange(0, 1_200_000, 10_000, dtype=np.int64)
    frame_ts = np.asarray(
        [
            700_000,
            800_000,
            900_000,
            1_000_000 + sync_offsets_us[0],
            1_100_000 + sync_offsets_us[1],
        ],
        dtype=np.int64,
    )
    with h5py.File(raw_path, "w") as handle:
        event = handle.create_group("prophesee/event_cam_left")
        event.create_dataset("x", data=np.full(len(event_t), 500, dtype=np.int16))
        event.create_dataset("y", data=np.full(len(event_t), 300, dtype=np.int16))
        event.create_dataset("t", data=event_t)
        event.create_dataset("p", data=np.arange(len(event_t), dtype=np.int8) % 2)
        milliseconds = np.arange(0, 1_202, dtype=np.int64)
        event.create_dataset("ms_map_idx", data=np.searchsorted(event_t, milliseconds * 1000))
        blackfly = handle.create_group("blackflys/left")
        blackfly.create_dataset("ts", data=frame_ts)
        frames = np.stack(
            [
                np.full((720, 1280, 3), 5, dtype=np.uint8),
                np.full((720, 1280, 3), 64, dtype=np.uint8),
                np.full((720, 1280, 3), 192, dtype=np.uint8),
                np.full((720, 1280, 3), 96, dtype=np.uint8),
                np.full((720, 1280, 3), 224, dtype=np.uint8),
            ]
        )
        blackfly.create_dataset("data", data=frames)

    sources = []
    for index in (1, 2):
        path = tmp_path / f"{index:04d}.json"
        path.write_text(
            json.dumps(
                {
                    "objects": [
                        {
                            "category": "car",
                            "group": 1,
                            "bbox": [450 + index, 250, 550 + index, 350],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        sources.append(
            {
                "available_us": int(frame_ts[index] + 50_000),
                "category": "car",
                "frame_index": index,
                "group": 1,
                "object_id": None,
                "path": str(path),
                "sha256": _sha256_file(path),
                "timestamp_us": int(frame_ts[index]),
            }
        )
    row = {
        "sequence_id": "fixture",
        "query_id": "fixture:00",
        "anchor_us": 1_100_000,
        "raw_path": str(raw_path),
        "raw_stat": {
            "size_bytes": raw_path.stat().st_size,
            "mtime_ns": raw_path.stat().st_mtime_ns,
        },
        "windows_us": [[800_000, 900_000], [900_000, 1_000_000], [1_000_000, 1_100_000]],
        "boxes_xyxy3": [[450, 250, 550, 350], [451, 250, 551, 350], [452, 250, 552, 350]],
        "square_xyxy": [400, 200, 600, 400],
        "bbox_sources": [sources[0], *sources],
        "scenario_family": "fixture",
        "speed_bucket": "medium",
        "target_type": "car",
    }
    row["metadata_sha256"] = _canonical_sha256(row)
    return row


def test_full_rgb_event_input_matches_published_shape_and_order(tmp_path: Path) -> None:
    result = prepare(_fixture(tmp_path, sync_offsets_us=(200, 300)))

    assert result["unavailable_reason"] is None
    assert result["sensor"].shape == (46, 128, 128)
    assert result["sensor"].dtype == np.float32
    assert result["garl_events"].shape == (40, 128, 128)
    assert np.array_equal(result["sensor"][6:], result["garl_events"])
    assert result["metadata"]["rgb_interval_us"] == 100_100
    assert result["metadata"]["rgb_frame_indices"] == [3, 4]
    assert result["metadata"]["source_bbox_frame_indices"] == [1, 2]
    assert result["metadata"]["rgb_event_endpoint_offsets_us"] == [200, 300]
    assert result["metadata"]["prediction_available_us"] == 1_100_300
    assert result["metadata"]["additional_sync_wait_us"] == 300
    assert result["metadata"]["available"] is True
    assert CONTRACT["fixed_dt_s"] == 0.1


def test_non100ms_rgb_pair_is_retained_as_unavailable_without_reselection(tmp_path: Path) -> None:
    result = prepare(_fixture(tmp_path, sync_offsets_us=(-1_000, 1_000)))

    assert result["sensor"] is None
    assert result["garl_events"].shape == (40, 128, 128)
    assert result["unavailable_reason"].startswith("RGB_ENDPOINT_INTERVAL_OUTSIDE_FIXED_DT")
    assert result["metadata"]["rgb_frame_indices"] == [3, 4]
    assert result["metadata"]["source_bbox_frame_indices"] == [1, 2]
    assert result["metadata"]["available"] is False


def test_nearest_sync_uses_earlier_frame_on_exact_tie_and_rejects_large_offset() -> None:
    tied = _nearest_frame_indices(
        np.asarray([999_000, 1_001_000, 1_100_000]),
        np.asarray([1_000_000, 1_100_000]),
    )
    missing = _nearest_frame_indices(
        np.asarray([990_000, 1_010_000, 1_100_000]),
        np.asarray([1_000_000, 1_100_000]),
    )

    assert tied is not None
    assert tied[0].tolist() == [0, 2]
    assert tied[1].tolist() == [-1_000, 0]
    assert missing is None


def test_frozen_bbox_hash_is_enforced(tmp_path: Path) -> None:
    row = _fixture(tmp_path, sync_offsets_us=(0, 0))
    Path(row["bbox_sources"][-1]["path"]).write_text("{}", encoding="utf-8")

    try:
        prepare(row)
    except ValueError as error:
        assert "bbox source SHA-256" in str(error)
    else:
        raise AssertionError("modified bbox source was accepted")
