"""Compare the local Garl preprocessing adapter with the frozen release code."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.eap import EAPEventReader  # noqa: E402
from e_jepa_ttc.data.garl_official_preprocessing import (  # noqa: E402
    official_resize_feature,
    official_square_box,
    official_timevolume_roi_np,
)
from e_jepa_ttc.data.garlttc_eap import (  # noqa: E402
    load_garlttc_train_index,
    normalize_boxes_xyxy,
    normalize_event_windows_us,
    resolve_eap_events_path,
)
from e_jepa_ttc.data.garlttc_lhr_cache import (  # noqa: E402
    _official_ttc_at_endpoint,
    select_temporal_indices,
)


def _select_rows(
    frame: pd.DataFrame, assignments: dict[str, list[str]], limit: int
) -> list[dict[str, object]]:
    allowed = [sequence for role in ("train", "validation") for sequence in assignments[role]]
    frame_any = cast(Any, frame)
    by_sequence: dict[str, list[dict[str, object]]] = {
        sequence: frame_any[frame_any["sequence_id"].astype(str) == sequence]
        .sort_values(["timestamp_us", "track_id", "sample_token"], kind="mergesort")
        .to_dict("records")
        for sequence in allowed
    }
    selected: list[Any] = []
    while len(selected) < limit:
        progressed = False
        for sequence in allowed:
            if by_sequence[sequence]:
                selected.append(by_sequence[sequence].pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def _as_list(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _official_resize(feature: np.ndarray, target_size: tuple[int, int]) -> torch.Tensor:
    from garl_ttc.datasets.ttc_dataset import (  # type: ignore[reportMissingImports]
        get_target_roi_from_feature_torch,
    )

    _, height, width = feature.shape
    return get_target_roi_from_feature_torch(
        torch.from_numpy(feature)[None],
        [0, 0, width, height],
        list(target_size),
    )[0]


def _rgb_shape(eap_root: Path, shard: object, member: object) -> tuple[int, int]:
    """Return the official RGB ``(width, height)`` sensor shape."""

    with tarfile.open(eap_root / str(shard), "r") as archive:
        extracted = archive.extractfile(str(member))
        if extracted is None:
            raise FileNotFoundError(f"{member} in {shard}")
        encoded = np.frombuffer(extracted.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV failed to decode {member} in {shard}")
    height, width = image_bgr.shape[:2]
    return int(width), int(height)


def run(args: argparse.Namespace) -> dict[str, Any]:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    assignments = split["assignments"]
    sequences = [sequence for role in ("train", "validation") for sequence in assignments[role]]
    index = load_garlttc_train_index(args.garlttc_root, sequences)
    rows = _select_rows(index.merged, assignments, args.samples)
    if len(rows) != args.samples:
        raise RuntimeError(f"Could only select {len(rows)} of {args.samples} parity rows.")

    reader_cache: dict[Path, EAPEventReader] = {}
    records: list[dict[str, Any]] = []
    for row in rows:
        boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
        windows = normalize_event_windows_us(row["event_windows_us"])
        frame_timestamps = [int(cast(Any, value)) for value in _as_list(row["frame_timestamps_us"])]
        first, second, _ = select_temporal_indices(
            frame_timestamps,
            anchor_timestamp_us=int(cast(Any, row["timestamp_us"])),
            target_delta_t_s=0.1,
            tolerance_s=0.025,
            context_delta_t_s=0.1,
            context_tolerance_s=0.05,
        )
        event_path = resolve_eap_events_path(args.eap_root, str(row["events_path"]))
        reader = reader_cache.setdefault(event_path, EAPEventReader(event_path))
        rgb_shards = _as_list(row["rgb_shard_paths"])
        rgb_members = _as_list(row["rgb_member_paths"])
        sensor_width, sensor_height = _rgb_shape(args.eap_root, rgb_shards[-1], rgb_members[-1])
        endpoint_metrics: list[dict[str, float | int]] = []
        for endpoint in (first, second):
            start_us, end_us = windows[endpoint]
            events = reader.read_window(start_us, end_us)
            square = official_square_box(boxes, endpoint)
            x = np.asarray(events["x"], dtype=np.int64) + 5
            y = np.asarray(events["y"], dtype=np.int64)
            timestamps = np.asarray(events["t"], dtype=np.int64)
            valid = (x >= 0) & (x < sensor_width) & (y >= 0) & (y < sensor_height)
            local_feature, local_counts = official_timevolume_roi_np(
                square, x[valid], y[valid], timestamps[valid]
            )
            from garl_ttc.datasets.event_representation import (  # type: ignore[reportMissingImports]
                get_timevolume_roi_np,
            )

            official_feature, official_counts = get_timevolume_roi_np(
                np.asarray(square, dtype=np.int16),
                x[valid].astype(np.int16),
                y[valid].astype(np.int16),
                timestamps[valid],
            )
            local_resized = official_resize_feature(local_feature, (128, 128))
            official_resized = _official_resize(official_feature, (128, 128))
            endpoint_metrics.append(
                {
                    "raw_max_abs": float(np.max(np.abs(local_feature - official_feature))),
                    "raw_count_max_abs": int(np.max(np.abs(local_counts - official_counts))),
                    "resized_max_abs": float(
                        torch.max(torch.abs(local_resized - official_resized)).item()
                    ),
                    "event_count": int(len(timestamps[valid])),
                }
            )
        records.append(
            {
                "sequence_id": str(row["sequence_id"]),
                "sample_token": str(row["sample_token"]),
                "ttc_target_s": float(_official_ttc_at_endpoint(row, second)),
                "delta_t_s": (frame_timestamps[second] - frame_timestamps[first]) * 1e-6,
                "sensor_width": sensor_width,
                "sensor_height": sensor_height,
                "endpoints": endpoint_metrics,
            }
        )
    raw_errors = [endpoint["raw_max_abs"] for row in records for endpoint in row["endpoints"]]
    resized_errors = [
        endpoint["resized_max_abs"] for row in records for endpoint in row["endpoints"]
    ]
    count_errors = [
        endpoint["raw_count_max_abs"] for row in records for endpoint in row["endpoints"]
    ]
    report = {
        "artifact_type": "garl_preprocessing_parity_v2",
        "release_root": args.release_root.as_posix(),
        "samples": len(records),
        "sequence_coverage": sorted({row["sequence_id"] for row in records}),
        "coverage_count": len({row["sequence_id"] for row in records}),
        "raw_max_abs": max(raw_errors, default=float("nan")),
        "resized_max_abs": max(resized_errors, default=float("nan")),
        "event_count_max_abs": max(count_errors, default=-1),
        "records": records,
        "status": "pass"
        if raw_errors and max(raw_errors) == 0.0 and max(count_errors) == 0
        else "fail",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    sys.path.insert(0, str(args.release_root.resolve()))
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
