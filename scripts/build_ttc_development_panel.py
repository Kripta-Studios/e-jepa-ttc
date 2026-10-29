"""Build a bounded public-TRAIN TTC panel without reading sensor payloads.

The panel binds Garl targets to exact eAP RGB members, records target-reference
versus input-availability timing, and declares readiness for four ROI/identity
conditions.  It never opens TEST assets or copies numeric TTC labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

JOIN_KEYS = ("sequence_id", "sample_token", "track_id", "public_track_id", "timestamp_us")


def sha256_file(path: Path) -> str:
    """Hash one admitted metadata source incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
    name: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    """Index exact metadata keys while refusing ambiguous duplicate rows."""
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in result:
            raise ValueError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def media_for_sequence(
    rows: Sequence[Mapping[str, Any]], sequence_id: str
) -> list[Mapping[str, Any]]:
    """Return only media owned by one query sequence."""
    result = [row for row in rows if str(row["sequence_id"]) == sequence_id]
    if not result:
        raise ValueError(f"No public train media found for query sequence: {sequence_id}")
    return result


def _box(box: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in box)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("ROI must contain four finite coordinates")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("ROI must have positive extent")
    return values


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = _box(left), _box(right)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def evaluator_assignment(
    predicted: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    *,
    minimum_iou: float = 0.3,
    maximum_center_distance_ratio: float = 0.5,
    ambiguity_margin: float = 0.05,
) -> dict[str, Any]:
    """Evaluation-only Hungarian ROI matching with bidirectional ambiguity.

    No identity, class, TTC, future frame or model score participates.
    """
    if not 0 <= minimum_iou <= 1 or maximum_center_distance_ratio < 0:
        raise ValueError("Matching thresholds are outside their domains")
    pboxes, rboxes = [_box(box) for box in predicted], [_box(box) for box in reference]
    cost = np.full((len(rboxes), len(pboxes)), np.inf, dtype=np.float64)
    for ri, target in enumerate(rboxes):
        diagonal = math.hypot(target[2] - target[0], target[3] - target[1])
        target_center = ((target[0] + target[2]) / 2, (target[1] + target[3]) / 2)
        for pi, proposal in enumerate(pboxes):
            overlap = _iou(target, proposal)
            proposal_center = ((proposal[0] + proposal[2]) / 2, (proposal[1] + proposal[3]) / 2)
            center_ratio = math.dist(target_center, proposal_center) / max(diagonal, 1e-12)
            if overlap >= minimum_iou and center_ratio <= maximum_center_distance_ratio:
                cost[ri, pi] = (1.0 - overlap) + 0.1 * center_ratio
    ambiguous_reference: set[int] = set()
    ambiguous_predicted: set[int] = set()
    for ri, row in enumerate(cost):
        finite = np.sort(row[np.isfinite(row)])
        if len(finite) > 1 and finite[1] - finite[0] < ambiguity_margin:
            ambiguous_reference.add(ri)
    for pi, column in enumerate(cost.T):
        finite = np.sort(column[np.isfinite(column)])
        if len(finite) > 1 and finite[1] - finite[0] < ambiguity_margin:
            ambiguous_predicted.add(pi)
    pairs = []
    if cost.size and np.isfinite(cost).any():
        rows, columns = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
        for ri, pi in zip(rows.tolist(), columns.tolist(), strict=True):
            if (
                np.isfinite(cost[ri, pi])
                and ri not in ambiguous_reference
                and pi not in ambiguous_predicted
            ):
                pairs.append(
                    {"reference_index": ri, "predicted_index": pi, "cost": float(cost[ri, pi])}
                )
    return {
        "pairs": pairs,
        "ambiguous_reference_indices": sorted(ambiguous_reference),
        "ambiguous_predicted_indices": sorted(ambiguous_predicted),
    }


def prediction_timing(
    query: Mapping[str, Any], target_media: Mapping[str, Any], previous_media: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the label reference to sensor availability without shifting TTC."""
    frame_times = [int(value) for value in query["frame_timestamps_us"]]
    windows = [[int(value) for value in window] for window in query["event_windows_us"]]
    target_index = len(frame_times) - 1
    if target_index < 1 or len(windows) != len(frame_times):
        raise ValueError("Garl query needs aligned causal endpoint arrays")
    exposure_start = int(target_media["rgb_exposure_start_timestamp_us"])
    exposure_end = int(target_media["rgb_exposure_end_timestamp_us"])
    if windows[target_index][1] != exposure_start:
        raise ValueError("Garl target event-window end must equal eAP RGB exposure start")
    if int(query["timestamp_us"]) != frame_times[target_index]:
        raise ValueError("Garl query timestamp must equal its final frame timestamp")
    previous_end = int(previous_media["rgb_exposure_end_timestamp_us"])
    if previous_end > exposure_start:
        raise ValueError("Previous RGB frame is not fully available at target reference")
    if exposure_end < exposure_start:
        raise ValueError("RGB exposure end precedes its start")
    return {
        "garl_label_clock_timestamp_us": frame_times[target_index],
        "sensor_reference_us": exposure_start,
        "sensor_reference_semantics": "garl_event_window_end_equals_rgb_exposure_start",
        "rgb_exposure_midpoint_us": (exposure_start + exposure_end) / 2.0,
        "prediction_available_cutoff_us": exposure_end,
        "label_age_at_prediction_us": exposure_end - exposure_start,
        "current_rgb_fully_available_at_reference": exposure_end <= exposure_start,
        "current_rgb_fully_available_at_prediction_cutoff": True,
        "previous_fully_available_frame": str(previous_media["sample_token"]),
        "previous_frame_available_us": previous_end,
        "evaluation_policy": "availability_lag_no_ttc_shift",
        "ttc_target_shift_applied_us": 0,
        "causality_statement": (
            "Inputs stop at prediction_available_cutoff_us. Current RGB contains an exposure "
            "that ends after the GT sensor reference, so the unchanged GT is evaluated with "
            "the recorded positive label age; strict same-instant causality would require "
            "excluding current RGB and using previous_fully_available_frame."
        ),
    }


def causal_prefix(
    frames: Sequence[Mapping[str, Any]], target_token: str, minimum_history_us: int
) -> list[Mapping[str, Any]]:
    """Return the shortest complete-frame suffix spanning the required history."""
    ordered = sorted(frames, key=lambda row: int(row["rgb_exposure_end_timestamp_us"]))
    positions = [index for index, row in enumerate(ordered) if row["sample_token"] == target_token]
    if len(positions) != 1:
        raise ValueError("Target token must identify exactly one public train frame")
    target_position = positions[0]
    cutoff = int(ordered[target_position]["rgb_exposure_end_timestamp_us"])
    start = target_position
    while (
        start > 0
        and cutoff - int(ordered[start]["rgb_exposure_end_timestamp_us"]) < minimum_history_us
    ):
        start -= 1
    result = ordered[start : target_position + 1]
    if cutoff - int(result[0]["rgb_exposure_end_timestamp_us"]) < minimum_history_us:
        raise ValueError("Public train frames do not provide the required causal history")
    if any(int(row["rgb_exposure_end_timestamp_us"]) > cutoff for row in result):
        raise ValueError("Causal prefix includes a future frame")
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Build the five-query panel and declare every missing dependency."""
    sources = {
        "builder": Path(__file__),
        "garl_train_inputs": args.garl_root / "data/train.parquet",
        "garl_train_annotations": args.garl_root / "annotations/train.parquet",
        "eap_train_media": args.eap_root / "data/train.parquet",
        "seed_detector_predictions": args.predictions,
    }
    if any(not path.is_file() for path in sources.values()):
        raise FileNotFoundError("A required public-train metadata source is missing")
    if any("test" in path.name.lower() for name, path in sources.items() if name != "builder"):
        raise ValueError("TEST paths are forbidden")
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = payload.get("predictions", payload.get("results"))
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("Seed detector artifact is empty")
    seed_tokens = set(predictions)
    seed_media = pq.read_table(
        sources["eap_train_media"],
        columns=["sample_token", "split", "sequence_id", "rgb_member_path"],
        filters=[("sample_token", "in", sorted(seed_tokens))],
        use_threads=False,
    ).to_pylist()
    if {row["split"] for row in seed_media} != {"train"}:
        raise ValueError("Only public train media are admitted")
    sequences = sorted({str(row["sequence_id"]) for row in seed_media})
    members = {(str(row["sequence_id"]), str(row["rgb_member_path"])) for row in seed_media}

    garl = pq.read_table(
        sources["garl_train_inputs"],
        columns=[
            *JOIN_KEYS,
            "frame_timestamps_us",
            "rgb_member_paths",
            "boxes_xyxy",
            "event_windows_us",
        ],
        filters=[("sequence_id", "in", sequences)],
        use_threads=False,
    ).to_pylist()
    candidates = [
        row
        for row in garl
        if any(
            (str(row["sequence_id"]), str(member)) in members for member in row["rgb_member_paths"]
        )
    ]
    candidates.sort(key=lambda row: (int(row["timestamp_us"]), str(row["sample_token"])))
    if not 1 <= len(candidates) <= args.maximum_queries:
        raise ValueError("Seed overlap does not produce the bounded query population")

    sequence_media = pq.read_table(
        sources["eap_train_media"],
        columns=[
            "sample_token",
            "split",
            "sequence_id",
            "rgb_member_path",
            "rgb_exposure_start_timestamp_us",
            "rgb_exposure_end_timestamp_us",
            "events_path",
            "rgb_shard_path",
        ],
        filters=[("sequence_id", "in", sequences)],
        use_threads=False,
    ).to_pylist()
    if any(row["split"] != "train" for row in sequence_media):
        raise ValueError("Sequence media escaped public TRAIN")
    media_by_member = _unique_index(
        sequence_media, ("sequence_id", "rgb_member_path"), "sequence/member media"
    )

    annotation_tokens = [str(row["sample_token"]) for row in candidates]
    annotations = pq.read_table(
        sources["garl_train_annotations"],
        columns=[*JOIN_KEYS, "ttc", "frame_ttc"],
        filters=[("sample_token", "in", annotation_tokens)],
        use_threads=False,
    ).to_pylist()
    annotation_by_key = _unique_index(annotations, JOIN_KEYS, "five-key TRAIN annotation")

    roi_by_track_member: dict[tuple[str, str, str], tuple[float, float, float, float]] = {}
    roi_conflicts: set[tuple[str, str, str]] = set()
    for row in garl:
        for member, box in zip(row["rgb_member_paths"], row["boxes_xyxy"], strict=True):
            key = (str(row["sequence_id"]), str(row["public_track_id"]), str(member))
            value = _box(box)
            if key in roi_by_track_member and roi_by_track_member[key] != value:
                roi_conflicts.add(key)
            roi_by_track_member[key] = value
    if roi_conflicts:
        raise ValueError("Reference ROI differs for the same sequence/track/member")

    panel = []
    for query in candidates:
        key = tuple(query[field] for field in JOIN_KEYS)
        annotation = annotation_by_key.get(key)
        if annotation is None:
            raise ValueError("A selected query lacks an exact five-key TRAIN annotation")
        frame_ttc = annotation["frame_ttc"]
        target_index = len(query["frame_timestamps_us"]) - 1
        if (
            target_index < 1
            or len(frame_ttc) <= target_index
            or not np.isfinite(float(frame_ttc[target_index]))
            or float(annotation["ttc"]) != float(frame_ttc[target_index])
        ):
            raise ValueError("Selected query lacks exact finite final-endpoint TTC")
        sequence = str(query["sequence_id"])
        target_member = str(query["rgb_member_paths"][target_index])
        target_media = media_by_member[(sequence, target_member)]
        query_sequence_media = media_for_sequence(sequence_media, sequence)
        prefix = causal_prefix(
            query_sequence_media, str(target_media["sample_token"]), args.minimum_history_us
        )
        previous = [
            row
            for row in query_sequence_media
            if int(row["rgb_exposure_end_timestamp_us"])
            <= int(target_media["rgb_exposure_start_timestamp_us"])
        ]
        if not previous:
            raise ValueError("No fully available RGB frame precedes the target reference")
        previous_media = max(previous, key=lambda row: int(row["rgb_exposure_end_timestamp_us"]))
        timing = prediction_timing(query, target_media, previous_media)
        track = str(query["public_track_id"])
        missing_reference_roi = [
            str(row["sample_token"])
            for row in prefix
            if (sequence, track, str(row["rgb_member_path"])) not in roi_by_track_member
        ]
        missing_detector = [
            str(row["sample_token"]) for row in prefix if row["sample_token"] not in seed_tokens
        ]
        reference_complete = not missing_reference_roi
        detector_complete = not missing_detector
        conditions = {
            "R0_reference_roi_reference_id": {
                "ready": reference_complete,
                "missing": missing_reference_roi,
            },
            "R1_perturbed_roi_reference_id": {
                "ready": reference_complete,
                "missing": missing_reference_roi,
            },
            "R2_detected_roi_reference_id": {
                "ready": reference_complete and detector_complete,
                "missing_reference_roi": missing_reference_roi,
                "missing_detector_prediction": missing_detector,
            },
            "R3_fully_predicted": {
                "ready": detector_complete,
                "missing_detector_prediction": missing_detector,
            },
        }
        panel.append(
            {
                "sequence_id": sequence,
                "garl_query_token": str(query["sample_token"]),
                "garl_public_track_id_evaluation_only": track,
                "target_frame_token": str(target_media["sample_token"]),
                "target_rgb_member_path": target_member,
                "target_reference_roi_xyxy_evaluation_only": list(
                    roi_by_track_member[(sequence, track, target_member)]
                ),
                "target_annotation_join_key": [query[field] for field in JOIN_KEYS],
                "target_ttc_present_not_materialized": True,
                "selection_evidence": {
                    "any_rgb_member_intersects_seed_replay": any(
                        (sequence, str(member)) in members for member in query["rgb_member_paths"]
                    ),
                    "target_rgb_member_intersects_seed_replay": (
                        sequence,
                        target_member,
                    )
                    in members,
                },
                "timing": timing,
                "prefix": {
                    "frame_tokens": [str(row["sample_token"]) for row in prefix],
                    "rgb_members": [str(row["rgb_member_path"]) for row in prefix],
                    "first_available_us": int(prefix[0]["rgb_exposure_end_timestamp_us"]),
                    "cutoff_us": int(target_media["rgb_exposure_end_timestamp_us"]),
                    "span_us": int(target_media["rgb_exposure_end_timestamp_us"])
                    - int(prefix[0]["rgb_exposure_end_timestamp_us"]),
                    "minimum_required_us": args.minimum_history_us,
                    "future_frames": 0,
                },
                "conditions": conditions,
            }
        )

    return {
        "schema": "ttc_development_five_query_panel_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PREPARED_CPU_NO_INFERENCE_DEVELOPMENT_EXPOSED",
        "scope": "public_train_metadata_and_labels_only_private_test_forbidden",
        "private_test_labels_read": False,
        "numeric_ttc_materialized": False,
        "sources": {name: _source(path) for name, path in sources.items()},
        "selection": {
            "rule": (
                "all Garl TRAIN queries for which any RGB member intersects the eight-frame "
                "seed replay; the target member itself need not intersect"
            ),
            "maximum_queries": args.maximum_queries,
            "selected_queries": len(panel),
            "selection_uses_ttc_value": False,
            "queries_with_target_member_outside_seed_replay": sum(
                not query["selection_evidence"]["target_rgb_member_intersects_seed_replay"]
                for query in panel
            ),
        },
        "timing_contract": {
            "target_reference": "Garl target event-window end == eAP RGB exposure start",
            "prediction_cutoff": "eAP RGB exposure end (current frame fully available)",
            "ttc_shift": "none",
            "strict_same_instant_alternative": (
                "exclude current RGB; use prior fully available frame"
            ),
        },
        "matching_preregistration": {
            "scope": "evaluator_only_never_runtime_input_for_R3",
            "coordinate_space": "eAP event image xyxy, width=1280, height=720",
            "algorithm": "global Hungarian on ROI cost",
            "cost": "(1-IoU)+0.1*(center_distance/reference_diagonal)",
            "minimum_iou": 0.3,
            "maximum_center_distance_over_reference_diagonal": 0.5,
            "bidirectional_ambiguity_margin": 0.05,
            "uses_class": False,
            "uses_identity": False,
            "uses_ttc": False,
            "unmatched_policy": "retain miss or false positive; never silently drop",
        },
        "runtime_tracking_preregistration": {
            "scope": "R3 causal inference only",
            "class_gate": "exact detector class equality",
            "minimum_iou": 0.3,
            "bidirectional_ambiguity_margin": 0.05,
            "confirm_hits": 2,
            "lost_ttl_frames": 2,
            "max_gap_us": 150_000,
            "uses_reference_identity_or_ttc": False,
        },
        "condition_contract": {
            "R0": "reference ROI + reference ID; oracle diagnostic",
            "R1": (
                "reference ROI translated by +0.05*width and +0.05*height with unchanged "
                "extent, shifted inward only to remain inside 1280x720; reference ID; "
                "oracle diagnostic"
            ),
            "R2": "detected ROI + evaluator-assigned reference ID; oracle identity diagnostic",
            "R3": "detected ROI + causal generated ID; GT matching only after inference",
            "lifetime": "fresh service/history/cache per condition and query",
        },
        "queries": panel,
        "readiness_summary": {
            condition: sum(bool(query["conditions"][condition]["ready"]) for query in panel)
            for condition in panel[0]["conditions"]
        },
        "next_action": (
            "Generate label-free detector predictions for every declared missing prefix frame, "
            "then execute the four isolated conditions only after a separate review."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garl-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-queries", type=int, default=5)
    parser.add_argument("--minimum-history-us", type=int, default=600_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path")
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "readiness": result["readiness_summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
