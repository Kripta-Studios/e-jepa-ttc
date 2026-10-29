"""Inspect public train correspondences without approving a geometry conversion.

Join eAP media and GarlTTC by sequence and RGB member, never by assuming that
annotation timestamps share the media clock. No test annotations or media are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    """Hash an input metadata file without loading its bytes at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    """Convert nested Parquet arrays, requiring a finite matrix of known shape."""
    result = np.stack([np.asarray(row, dtype=np.float64) for row in value])
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite matrix {shape}, received {result.shape}")
    return result


def build_media_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict]:
    """Index unique train media identities; refuse an ambiguous correspondence."""
    result = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Only public train media are admitted")
        key = (str(row["sequence_id"]), str(row["rgb_member_path"]))
        if key in result:
            raise ValueError(f"Ambiguous RGB member identity: {key}")
        result[key] = dict(row)
    return result


def match_endpoint(
    media: Mapping[tuple[str, str], Mapping[str, Any]],
    query: Mapping[str, Any],
    endpoint: int,
) -> dict[str, Any]:
    """Match by explicit media identity and measure, rather than erase, clock offsets."""
    key = (str(query["sequence_id"]), str(query["rgb_member_paths"][endpoint]))
    if key not in media:
        raise ValueError(f"No train media match for {key}")
    row = dict(media[key])
    if row["rgb_shard_path"] != query["rgb_shard_paths"][endpoint]:
        raise ValueError(f"Shard differs for RGB identity {key}")
    if row["events_path"] != query["events_path"]:
        raise ValueError(f"Event stream differs for RGB identity {key}")
    annotation_us = int(query["frame_timestamps_us"][endpoint])
    start_us = int(row["rgb_exposure_start_timestamp_us"])
    end_us = int(row["rgb_exposure_end_timestamp_us"])
    if end_us < start_us:
        raise ValueError("Exposure end precedes exposure start")
    window = [int(t) for t in query["event_windows_us"][endpoint]]
    if len(window) != 2 or window[1] < window[0]:
        raise ValueError("Invalid event interval")
    return {
        "media": row,
        "annotation_timestamp_us": annotation_us,
        "exposure_start_us": start_us,
        "exposure_end_us": end_us,
        "annotation_minus_exposure_start_us": annotation_us - start_us,
        "event_window_us": window,
        "event_window_end_minus_exposure_start_us": window[1] - start_us,
    }


def projection_diagnostic(
    corners: Any,
    intrinsic: Any,
    transform: Any,
    roi: Any,
    box3d_h: float,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Measure reprojection and edge lengths, without naming HF dimensions."""
    camera = numeric_matrix(corners, (8, 3))
    k_event = numeric_matrix(intrinsic, (3, 3))
    event_from_ego = numeric_matrix(transform, (4, 4))
    if np.any(camera[:, 2] <= 0):
        raise ValueError("At least one stored camera corner is behind the image plane")
    target_roi = np.asarray(roi, dtype=np.float64)
    if target_roi.shape != (4,) or not np.isfinite(target_roi).all():
        raise ValueError("ROI must have four finite coordinates")
    if np.any(target_roi[2:] <= target_roi[:2]):
        raise ValueError("ROI extent must be positive")
    if not np.isfinite(box3d_h) or box3d_h <= 0:
        raise ValueError("box3d_h must be finite and positive")
    if min(image_size) <= 0:
        raise ValueError("Image size must be positive")
    projected = camera @ k_event.T
    if np.any(projected[:, 2] == 0):
        raise ValueError("Projection denominator is zero")
    pixels = projected[:, :2] / projected[:, 2:3]
    raw = np.concatenate((pixels.min(axis=0), pixels.max(axis=0)))
    limits = np.asarray([image_size[0], image_size[1]] * 2)
    clipped = np.clip(raw, 0, limits)
    pixel_clipped = np.clip(raw, 0, limits - 1)
    inverse = np.linalg.inv(event_from_ego)
    homogeneous = np.column_stack((camera, np.ones(8))) @ inverse.T
    if np.any(homogeneous[:, 3] == 0):
        raise ValueError("Inverse transform produces a point at infinity")
    ego = homogeneous[:, :3] / homogeneous[:, 3:4]
    z_extent = float(np.ptp(ego[:, 2]))
    return {
        "raw_projected_bbox_xyxy": raw.tolist(),
        "clipped_projected_bbox_xyxy": clipped.tolist(),
        "raw_max_abs_roi_error_px": float(np.max(np.abs(raw - target_roi))),
        "clipped_max_abs_roi_error_px": float(np.max(np.abs(clipped - target_roi))),
        "pixel_center_clipped_max_abs_roi_error_px": float(
            np.max(np.abs(pixel_clipped - target_roi))
        ),
        "clipping_applied": bool(np.any(raw != clipped)),
        "camera_edge_lengths_01_03_04_m": [
            float(np.linalg.norm(camera[j] - camera[0])) for j in (1, 3, 4)
        ],
        "ego_z_extent_under_manifest_transform_m": z_extent,
        "garl_box3d_h_m": float(box3d_h),
        "box3d_h_minus_ego_z_extent_m": float(box3d_h - z_extent),
    }


def original_bundle_hypothesis(box: Any) -> np.ndarray:
    """Apply the inspected original-bundle rule as a hypothesis, not an HF adapter.

    Matches geometry_contract.original_bundle_corners: xyzlhwyaw, local X width,
    local Y length, local Z height, clockwise yaw. Only a corner-set comparison
    is used below because corner order is not established across releases.
    """
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
        raise ValueError("HF box must have seven finite values and positive dimensions")
    signs = np.asarray(
        [
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
        ]
    )
    c, s = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return (signs * box[[5, 3, 4]] / 2) @ rotation.T + box[:3]


def compare_hf_frame(
    camera_corners: Any, transform: Any, rows: Sequence[Mapping[str, Any]], track_id: str
) -> dict[str, Any]:
    """Test ID equality and nearest center separately; neither establishes identity."""
    camera = numeric_matrix(camera_corners, (8, 3))
    event_from_ego = numeric_matrix(transform, (4, 4))
    ego_h = np.column_stack((camera, np.ones(8))) @ np.linalg.inv(event_from_ego).T
    ego = ego_h[:, :3] / ego_h[:, 3:4]
    center = ego.mean(axis=0)
    identities = [str(row["instance_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate HF instance_id within matched frame")

    def compare(row: Mapping[str, Any]) -> dict[str, Any]:
        corners = original_bundle_hypothesis(row["bbox_3d_ego"])
        distances = np.linalg.norm(corners[:, None, :] - ego[None, :, :], axis=2)
        return {
            "instance_id": str(row["instance_id"]),
            "category": str(row["category"]),
            "center_distance_m": float(np.linalg.norm(corners.mean(axis=0) - center)),
            "corner_set_hausdorff_m": float(max(distances.min(0).max(), distances.min(1).max())),
        }

    comparisons = [compare(row) for row in rows]
    nearest = min(
        comparisons, key=lambda c: (c["center_distance_m"], c["instance_id"]), default=None
    )
    same_id = next((c for c in comparisons if c["instance_id"] == track_id), None)
    return {
        "garl_public_track_id": track_id,
        "garl_corner_center_ego_m": center.tolist(),
        "hf_frame_annotation_count": len(rows),
        "same_id_hypothesis": same_id,
        "nearest_center_diagnostic_not_identity": nearest,
    }


def audit(eap_root: Path, garl_root: Path, per_sequence: int) -> dict[str, Any]:
    """Audit a deterministic temporal spread of train queries in every sequence."""
    if per_sequence < 1:
        raise ValueError("per_sequence must be positive")
    sources = {
        "eap_train_media": eap_root / "data/train.parquet",
        "garl_train_inputs": garl_root / "data/train.parquet",
        "garl_train_annotations": garl_root / "annotations/train.parquet",
    }
    tables = {name: pd.read_parquet(path) for name, path in sources.items()}
    media_table = tables["eap_train_media"]
    inputs = tables["garl_train_inputs"]
    annotations = tables["garl_train_annotations"]
    for name, table in tables.items():
        if table.sample_token.isna().any() or table.sample_token.duplicated().any():
            raise ValueError(f"Missing or duplicate sample_token in {name}")
    media = build_media_index(media_table.to_dict("records"))
    labels = {row["sample_token"]: row for row in annotations.to_dict("records")}
    if set(inputs.sample_token) != set(labels):
        raise ValueError("Garl train inputs and annotation populations differ")
    records = []
    errors = []
    selected = []
    for _, group in inputs.groupby("sequence_id", sort=True):
        group = group.sort_values(["timestamp_us", "sample_token"])
        positions = np.linspace(0, len(group) - 1, min(per_sequence, len(group)), dtype=int)
        selected.extend(group.iloc[positions].to_dict("records"))
    hf_frames = {}
    for sequence in sorted({q["sequence_id"] for q in selected}):
        path = eap_root / "data/train" / sequence / "labels.parquet"
        sources[f"eap_labels_{sequence}"] = path
        frame_tokens = {
            media[(sequence, member)]["sample_token"]
            for q in selected
            if q["sequence_id"] == sequence
            for member in q["rgb_member_paths"]
        }
        table = pd.read_parquet(
            path,
            columns=[
                "sample_token",
                "split",
                "sequence_id",
                "instance_id",
                "category",
                "bbox_3d_ego",
            ],
        )
        if not (table["split"].eq("train") & table["sequence_id"].eq(sequence)).all():
            raise ValueError(f"Unexpected split or sequence in {path}")
        for token, frame in table[table.sample_token.isin(frame_tokens)].groupby("sample_token"):
            hf_frames[token] = frame.to_dict("records")
    for query in selected:
        label = labels[query["sample_token"]]
        if label["sequence_id"] != query["sequence_id"]:
            raise ValueError("Input/label sequence mismatch")
        count = len(query["rgb_member_paths"])
        if count != 2:
            raise ValueError("This diagnostic requires the released two-endpoint Garl contract")
        arrays = [
            query["frame_timestamps_us"],
            query["rgb_shard_paths"],
            query["event_windows_us"],
            query["boxes_xyxy"],
            label["box3d_Fcam"],
        ]
        if any(len(values) != count for values in arrays):
            raise ValueError("Endpoint arrays differ in length")
        for endpoint in range(count):
            identity = {
                "sequence_id": str(query["sequence_id"]),
                "query_token": str(query["sample_token"]),
                "endpoint": endpoint,
                "rgb_member_path": str(query["rgb_member_paths"][endpoint]),
            }
            try:
                match = match_endpoint(media, query, endpoint)
                mr = match.pop("media")
                geometry = projection_diagnostic(
                    label["box3d_Fcam"][endpoint],
                    mr["K_event"],
                    mr["T_event_ego"],
                    query["boxes_xyxy"][endpoint],
                    float(label["box3d_h"]),
                    (1280, 720),
                )
                hf = compare_hf_frame(
                    label["box3d_Fcam"][endpoint],
                    mr["T_event_ego"],
                    hf_frames.get(mr["sample_token"], []),
                    str(query["public_track_id"]),
                )
                records.append(
                    {**identity, "eap_token": mr["sample_token"], **match, **geometry, **hf}
                )
            except (ValueError, KeyError, np.linalg.LinAlgError) as error:
                errors.append({**identity, "error": str(error)})
    by_sequence = {}
    for sequence in sorted(set(inputs.sequence_id)):
        subset = [r for r in records if r["sequence_id"] == sequence]
        offsets = [r["annotation_minus_exposure_start_us"] for r in subset]
        by_sequence[sequence] = {
            "diagnosed_endpoints": len(subset),
            "clock_offset_min_us": min(offsets) if offsets else None,
            "clock_offset_max_us": max(offsets) if offsets else None,
        }
    return {
        "schema": "eap_garl_alignment_diagnostic_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "DIAGNOSTIC_ONLY_G0_BLOCKED",
        "script_sha256": sha256_file(Path(__file__)),
        "sources": {
            name: {"path": str(p), "sha256": sha256_file(p)} for name, p in sources.items()
        },
        "selection": "Evenly spaced positions after timestamp/token sort, separately per sequence",
        "queries_per_sequence_requested": per_sequence,
        "queries_selected": len(selected),
        "sequences_selected": len(by_sequence),
        "endpoints_diagnosed": len(records),
        "endpoint_errors": errors,
        "summary": {
            "sequences_with_nonconstant_sampled_clock_offset": sum(
                item["clock_offset_min_us"] != item["clock_offset_max_us"]
                for item in by_sequence.values()
            ),
            "different_annotation_and_media_clocks": sum(
                r["annotation_minus_exposure_start_us"] != 0 for r in records
            ),
            "event_window_end_equals_exposure_start": sum(
                r["event_window_end_minus_exposure_start_us"] == 0 for r in records
            ),
            "clipped_projection_roi_error_below_one_pixel": sum(
                r["clipped_max_abs_roi_error_px"] < 1.0 for r in records
            ),
            "clipped_projection_roi_error_max_px": max(
                (r["clipped_max_abs_roi_error_px"] for r in records), default=None
            ),
            "pixel_center_clipped_projection_roi_error_below_one_pixel": sum(
                r["pixel_center_clipped_max_abs_roi_error_px"] < 1.0 for r in records
            ),
            "pixel_center_clipped_projection_roi_error_max_px": max(
                (r["pixel_center_clipped_max_abs_roi_error_px"] for r in records), default=None
            ),
            "same_id_available": sum(r["same_id_hypothesis"] is not None for r in records),
            "same_id_corners_within_1mm": sum(
                r["same_id_hypothesis"] is not None
                and r["same_id_hypothesis"]["corner_set_hausdorff_m"] <= 0.001
                for r in records
            ),
            "nearest_center_has_same_id": sum(
                r["nearest_center_diagnostic_not_identity"] is not None
                and r["nearest_center_diagnostic_not_identity"]["instance_id"]
                == r["garl_public_track_id"]
                for r in records
            ),
            "nearest_center_corners_within_1mm": sum(
                r["nearest_center_diagnostic_not_identity"] is not None
                and r["nearest_center_diagnostic_not_identity"]["corner_set_hausdorff_m"] <= 0.001
                for r in records
            ),
            "box3d_h_disagrees_with_target_endpoint_z_extent_over_1mm": sum(
                abs(r["box3d_h_minus_ego_z_extent_m"]) > 0.001
                for r in records
                if r["endpoint"] == 1
            ),
            "box3d_h_matches_target_endpoint_edge_01_within_1mm": sum(
                abs(r["garl_box3d_h_m"] - r["camera_edge_lengths_01_03_04_m"][0]) <= 0.001
                for r in records
                if r["endpoint"] == 1
            ),
        },
        "per_sequence": by_sequence,
        "records": records,
        "limitations": [
            "Public train metadata only; no original pkl or submission scorer inspected",
            "Projection agreement with stored ROIs does not prove RGB pixel rectification",
            "Image clipping compares both extent and last-pixel-center hypotheses at 1280x720",
            "Same-ID and nearest-center comparisons do not establish cross-release object identity",
            "Original-bundle corner rule is a tested hypothesis, not an approved HF adapter",
            "box3d_h describes target endpoint; compare it to endpoint 1, not prior endpoint",
            "No guarantee of physical clock mapping, arrival time or velocity semantics",
            "No data gate is approved by this diagnostic; G0 remains blocked",
        ],
    }


def main() -> int:
    """Write a diagnostic receipt while leaving the geometry gate blocked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garl-root", type=Path, required=True)
    parser.add_argument("--per-sequence", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.eap_root, args.garl_root, args.per_sequence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}))
    return 2 if report["endpoint_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
