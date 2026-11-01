"""Freeze a varied, sensor-only EvTTC comparison before inference and TTC scoring.

The historical author-interval cohort is retained. The additional cohort samples
up to sixteen chronological anchors per available sequence without reading TTC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from operational.evttc_transfer.inputs import (
    EvTTCEventReader,
    _calibration,
    _calibration_sha256,
    _canonical_sha256,
    _single_object_bbox,
    project_rgb_box_rotation_only,
)
from scripts.post_v18_transfer_inputs import ROOT, nearest, sha

DEFAULT_OUTPUT = ROOT / "artifacts/final_train40_transfer_20261027"
DEFAULT_MANIFEST = ROOT / "data/manifests/evttc_all32_local.yaml"
HISTORICAL = ROOT / "docs/post_v18_transfer_20261017/evidence/QUERY_MANIFEST.json"


def uniform_indices(size: int, limit: int | None = 16) -> list[int]:
    """Include both endpoints using deterministic integer-spaced indices."""
    if limit is None or size <= limit:
        return list(range(size))
    return [index * (size - 1) // (limit - 1) for index in range(limit)]


def immutable_json(path: Path, value: Any) -> None:
    """Resume identical registrations, rejecting any overwrite of frozen IDs."""
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen cohort differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def sequence_population(
    spec: dict[str, Any], data_parent: Path, *, limit: int | None = 16
) -> dict[str, Any]:
    """Read public ROI and sensor clocks only; never parse a TTC or depth asset."""
    folder = data_parent / spec["local_path"]
    raw = folder / spec["event_hdf5"]
    labels = folder / spec["label_dir"]
    sequence = str(spec["sequence_id"])
    target = str(spec["target_type"])
    excluded: list[dict[str, Any]] = []
    if not raw.is_file() or not labels.is_dir():
        return {
            "rows": [],
            "excluded": [{"sequence_id": sequence, "reason": "MISSING_SENSOR_OR_ROI"}],
        }
    stat = raw.stat()
    with h5py.File(raw, "r") as handle:
        clock = np.asarray(handle["blackflys/left/ts"], dtype=np.int64)
        calibration = _calibration(handle)
        with EvTTCEventReader(raw) as reader:
            event_clock = handle[reader.layout.t]
            event_start, event_end = int(event_clock[0]), int(event_clock[-1])
    if len(clock) < 3 or np.any(np.diff(clock) <= 0):
        raise ValueError(f"Invalid RGB clock: {raw}")
    frames: dict[int, dict[str, Any]] = {}
    for path in sorted(labels.glob("*.json")):
        if not path.stem.isdigit():
            continue
        index = int(path.stem)
        if not 0 <= index < len(clock):
            excluded.append(
                {"sequence_id": sequence, "frame_index": index, "reason": "ROI_FRAME_OUT_OF_RANGE"}
            )
            continue
        objects = json.loads(path.read_text(encoding="utf-8")).get("objects", [])
        candidates = [
            (i, item)
            for i, item in enumerate(objects)
            if str(item.get("category", "")).lower() == target
        ]
        # TTC is a single scenario-target series, not per-background-object GT.
        # Keep only the unambiguous primary annotation, never choose by TTC.
        if len(candidates) != 1 or candidates[0][1].get("group") != 1:
            excluded.append(
                {
                    "sequence_id": sequence,
                    "frame_index": index,
                    "reason": "AMBIGUOUS_OR_NONPRIMARY_SCENARIO_TARGET",
                    "matching_objects": len(candidates),
                }
            )
            continue
        object_index, obj = candidates[0]
        box = _single_object_bbox(obj)
        if box is None:
            excluded.append(
                {"sequence_id": sequence, "frame_index": index, "reason": "INVALID_ROI"}
            )
            continue
        event_box = project_rgb_box_rotation_only(box, calibration)
        frames[index] = {
            "index": index,
            "timestamp_us": int(clock[index]),
            "rgb_box": list(box),
            "event_box": list(event_box),
            "source": {
                "path": str(path.resolve()),
                "sha256": sha(path),
                "frame_index": index,
                "timestamp_us": int(clock[index]),
                "available_us": int(clock[index]),
                "category": obj["category"],
                "group": obj.get("group"),
                "object_index": object_index,
            },
        }
    candidates = []
    for index in sorted(frames):
        anchor = int(clock[index])
        ids = [nearest(clock, anchor - lag) for lag in (200_000, 100_000, 0)]
        reason = None
        if any(
            abs(int(clock[i]) - (anchor - lag)) > 1000
            for i, lag in zip(ids, (200_000, 100_000, 0), strict=True)
        ):
            reason = "NO_NATIVE_100MS_TRIPLET"
        elif any(i not in frames for i in ids):
            reason = "INCOMPLETE_TARGET_ROI_TRIPLET"
        elif anchor - 600_000 < event_start or anchor > event_end:
            reason = "INSUFFICIENT_FULL_600MS_EVENT_SUPPORT"
        if reason:
            excluded.append(
                {"sequence_id": sequence, "anchor_us": anchor, "stage": "query", "reason": reason}
            )
            continue
        triple = [frames[i] for i in ids]
        boxes = [item["event_box"] for item in triple]
        row = {
            "query_id": f"EXTENDED:{sequence}:{anchor}",
            "sequence_id": sequence,
            "physical_sequence_id": sequence,
            "anchor_us": anchor,
            "raw_path": str(raw.resolve()),
            "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "windows_us": [
                [anchor + offset - 100_000, anchor + offset] for offset in (-200_000, -100_000, 0)
            ],
            "boxes_xyxy3": boxes,
            "square_xyxy": common_square_from_boxes(boxes, (1, 2), margin_fraction=0.25),
            "bbox_sources": [item["source"] for item in triple],
            "rgb_boxes": [item["rgb_box"] for item in triple],
            "rgb_frame_indices": ids,
            "rgb_timestamps_us": [int(clock[i]) for i in ids],
            "calibration_sha256": _calibration_sha256(calibration),
            "multiple_car_objects": False,
            "target_type": target,
            "scenario_family": spec["scenario_family"],
            "speed_bucket": spec["speed_bucket"],
            "ttc_csv_path": str((folder / spec["ttc_csv"]).resolve()),
            "target_identity_policy": (
                "unique scenario target_type, annotation group1; scalar scenario TTC, "
                "not per-object GT"
            ),
            "strata": ["EXTENDED_UNIFORM_16"],
        }
        row["metadata_sha256"] = _canonical_sha256(row)
        candidates.append(row)
    chosen = [candidates[i] for i in uniform_indices(len(candidates), limit)]
    return {
        "rows": chosen,
        "excluded": excluded,
        "inventory": {
            "sequence_id": sequence,
            "raw_path": str(raw.resolve()),
            "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "target_type": target,
            "label_directory": str(labels.resolve()),
            "ttc_file_present": (folder / spec["ttc_csv"]).is_file(),
            "sensor_range_us": [event_start, event_end],
            "eligible_anchors": len(candidates),
            "selected_anchors": len(chosen),
            "eligible_anchor_us": [row["anchor_us"] for row in candidates],
        },
    }


def build(manifest_path: Path, data_parent: Path) -> dict[str, Any]:
    """Freeze the union; historical overlap is tagged and never double-counted."""
    specs = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["sequences"]
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    excluded, inventory = [], []
    by_raw = {}
    for spec in sorted(specs, key=lambda value: value["sequence_id"]):
        result = sequence_population(spec, data_parent)
        excluded.extend(result["excluded"])
        if "inventory" in result:
            inventory.append(result["inventory"])
        raw = str((data_parent / spec["local_path"] / spec["event_hdf5"]).resolve())
        by_raw[raw] = spec
        for row in result["rows"]:
            rows[(row["raw_path"], row["anchor_us"])] = row
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))["rows"]
    if len(historical) != 46:
        raise ValueError("Historical cohort must retain exactly 46 registered queries")
    overlap = 0
    for original in historical:
        row = dict(original)
        key = (row["raw_path"], row["anchor_us"])
        spec = by_raw[row["raw_path"]]
        strata = ["HISTORICAL_AUTHOR46"]
        if key in rows:
            strata.append("EXTENDED_UNIFORM_16")
            overlap += 1
        row.update(
            {
                "physical_sequence_id": spec["sequence_id"],
                "target_type": spec["target_type"],
                "scenario_family": spec["scenario_family"],
                "speed_bucket": spec["speed_bucket"],
                "strata": strata,
                "ttc_csv_path": str((data_parent / spec["local_path"] / spec["ttc_csv"]).resolve()),
            }
        )
        row.pop("metadata_sha256", None)
        row["metadata_sha256"] = _canonical_sha256(row)
        rows[key] = row
    return {
        "schema": "final_train40_transfer_cohort_v1",
        "rows": sorted(rows.values(), key=lambda r: (r["physical_sequence_id"], r["anchor_us"])),
        "inventory": inventory,
        "excluded": excluded,
        "historical_queries": 46,
        "overlap_physical_queries": overlap,
        "targets_read": False,
        "optimizer_updates": 0,
        "model_selection": False,
        "cohort_policy": (
            "up to16 integer-uniform chronological eligible anchors per sequence; "
            "preserve all46 historical author-interval queries; "
            "union deduplicated by raw path and anchor"
        ),
        "exposure": (
            "EvTTC observed in historical development; frozen TRAIN40 deployment has "
            "no EvTTC fitting, but this is descriptive transfer rather than blind confirmation"
        ),
        "fcwd": (
            "excluded: existing full multimodal adapter has no validated event-right to RGB mapping"
        ),
        "eap": "not in this cohort: all public TRAIN40 is deployment training data",
        "target_linkage_limit": (
            "Single TTC series per scenario; category and primary annotation group1 identify "
            "scenario target. No per-object TTC identifier is published in the five-column table. "
            "Ambiguous target ROI frames excluded before TTC access."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-parent", type=Path, default=ROOT.parent / "e-jepa-ttc")
    args = parser.parse_args()
    cohort = build(args.manifest.resolve(), args.data_parent.resolve())
    destination = args.output / "QUERY_MANIFEST.json"
    immutable_json(destination, cohort)
    registration = {
        "schema": "final_train40_transfer_cohort_registration_v1",
        "manifest_sha256": sha(destination),
        "source_manifest_sha256": sha(args.manifest),
        "historical_manifest_sha256": sha(HISTORICAL),
        "builder_sha256": sha(Path(__file__)),
        "queries": len(cohort["rows"]),
        "sequences": len(cohort["inventory"]),
        "targets_read": False,
        "optimizer_updates": 0,
        "seed_selection": False,
        "external_submission": False,
    }
    immutable_json(args.output / "COHORT_REGISTRATION.json", registration)
    print(json.dumps(registration))


if __name__ == "__main__":
    main()
