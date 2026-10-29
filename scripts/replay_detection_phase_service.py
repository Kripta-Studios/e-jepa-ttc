"""Prepare or execute a label-free temporal7 detector-to-TRAIN40 replay.

Preparation is CPU-only and verifies the sealed candidate, public train media,
causal timestamps, projected detector geometry and multi-object association.
Execution requires an explicitly selected CUDA device and performs one actual
MultiObjectTrain40Service update at the final sensor cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402

from e_jepa_ttc.detection.causal_tracker import (  # noqa: E402
    CausalMultiObjectTracker,
    DetectionObservation,
)
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash an admitted input incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Image-plane IoU for positive xyxy regions."""
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    a = (left[2] - left[0]) * (left[3] - left[1])
    b = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(a + b - intersection, 1e-12)


def project_detection(
    row: dict[str, Any],
    *,
    token: str,
    index: int,
    intrinsic: np.ndarray,
    event_from_ego: np.ndarray,
    minimum_score: float,
) -> DetectionObservation | None:
    """Project one label-free detector cuboid into the public event image."""
    forbidden = {"ground_truth", "annotations", "target", "ttc_true"}
    if forbidden & set(row):
        raise ValueError("Prediction rows must not carry targets or annotations")
    score = float(row["score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Detection score must be finite and in [0,1]")
    if score < minimum_score:
        return None
    center = tuple(float(value) for value in row["translation_m"])
    size = tuple(float(value) for value in row["size_lwh_m"])
    yaw = float(row["yaw_rad"])
    if len(center) != 3 or len(size) != 3 or any(value <= 0 for value in size):
        raise ValueError("Detector cuboid must use center xyz and positive size lwh")
    box = [*center, size[0], size[2], size[1], yaw]
    corners = original_bundle_corners(box)
    camera = (np.column_stack((corners, np.ones(8, dtype=np.float64))) @ event_from_ego.T)[:, :3]
    if np.any(camera[:, 2] <= 1e-4):
        return None
    projected = camera @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    clipped = np.clip(np.r_[pixels.min(0), pixels.max(0)], 0, [1279, 719, 1279, 719])
    if np.any(clipped[2:] - clipped[:2] < 8):
        return None
    velocity_value = row.get("velocity_mps")
    velocity = (
        tuple(float(value) for value in velocity_value) if velocity_value is not None else None
    )
    square = (float(clipped[0]), float(clipped[1]), float(clipped[2]), float(clipped[3]))
    return DetectionObservation(
        detection_id=f"{token}:{index}",
        class_name=str(row["class_name"]),
        score=score,
        event_square=square,
        rgb_square=square,
        center_3d_m=center,
        size_lwh_m=size,
        yaw_rad=yaw,
        velocity=velocity,
        velocity_frame="released_native_components" if velocity is not None else None,
        velocity_semantics=(
            "published_first_components_not_derived_ttc" if velocity is not None else None
        ),
    )


def suppress_duplicates(
    detections: list[DetectionObservation], threshold: float
) -> list[DetectionObservation]:
    """Apply class-aware NMS without editing the saved detector predictions."""
    result = []
    for item in sorted(detections, key=lambda value: (-value.score, value.detection_id)):
        if any(
            old.class_name == item.class_name
            and box_iou(old.event_square, item.event_square) > threshold
            for old in result
        ):
            continue
        result.append(item)
    return result


def build_frame_plan(
    predictions: dict[str, list[dict[str, Any]]],
    media: dict[str, dict[str, Any]],
    *,
    minimum_score: float,
    nms_iou: float,
) -> list[dict[str, Any]]:
    """Build chronological detector-only frames using public calibration."""
    if set(predictions) != set(media):
        raise ValueError("Predictions and admitted media tokens must match exactly")
    sequences = {str(row["sequence_id"]) for row in media.values()}
    if len(sequences) != 1:
        raise ValueError("Replay requires exactly one sequence")
    frames = []
    for token in sorted(
        media, key=lambda value: int(media[value]["rgb_exposure_end_timestamp_us"])
    ):
        row = media[token]
        start = int(row["rgb_exposure_start_timestamp_us"])
        end = int(row["rgb_exposure_end_timestamp_us"])
        if row["split"] != "train" or end < start:
            raise ValueError("Replay accepts ordered public-train exposures only")
        intrinsic = np.asarray(row["K_event"], dtype=np.float64)
        transform = np.asarray(row["T_event_ego"], dtype=np.float64)
        raw_detections = predictions[token]
        below_score = sum(float(item["score"]) < minimum_score for item in raw_detections)
        detections = []
        for index, detection in enumerate(raw_detections):
            projected = project_detection(
                detection,
                token=token,
                index=index,
                intrinsic=intrinsic,
                event_from_ego=transform,
                minimum_score=minimum_score,
            )
            if projected is not None:
                detections.append(projected)
        retained = suppress_duplicates(detections, nms_iou)
        frames.append(
            {
                "sample_token": token,
                "start_us": start,
                "end_us": end,
                "detections": retained,
                "filter_accounting": {
                    "raw_predictions": len(raw_detections),
                    "below_score": below_score,
                    "projection_or_minimum_size_rejected": (
                        len(raw_detections) - below_score - len(detections)
                    ),
                    "class_aware_nms_suppressed": len(detections) - len(retained),
                    "retained": len(retained),
                },
            }
        )
    if any(
        left["end_us"] >= right["end_us"]
        for left, right in zip(frames[:-1], frames[1:], strict=True)
    ):
        raise ValueError("Exposure endpoints must increase strictly")
    return frames


def replay_tracker_prefix(
    frames: list[dict[str, Any]], tracker: CausalMultiObjectTracker
) -> list[dict[str, Any]]:
    """Associate all frames before the final service-owned update."""
    receipts = []
    for frame in frames[:-1]:
        result = tracker.update(frame["end_us"], frame["detections"])
        receipts.append(
            {
                "sample_token": frame["sample_token"],
                "timestamp_us": frame["end_us"],
                "assignments": [
                    {
                        "detection_id": item.detection.detection_id,
                        "object_id": item.track.object_id,
                        "status": item.association_status,
                        "state": str(item.track.state),
                    }
                    for item in result.assignments
                ],
            }
        )
    return receipts


def causal_rgb_groups(timestamps: dict[int, int], query_us: int) -> list[tuple[int, ...]]:
    """Build three-frame groups wholly inside the frozen 650 ms history."""
    indices = sorted(timestamps)
    if indices != list(range(len(indices))):
        raise ValueError("RGB history indices must be contiguous from zero")
    return [
        tuple(range(index - 2, index + 1))
        for index in range(2, len(indices))
        if 0 <= query_us - timestamps[index - 2] <= 650_000
    ]


def _read_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    forbidden = {"ground_truth", "labels", "annotations", "target"}
    if forbidden & set(payload):
        raise ValueError("Use the label-free inference JSON")
    predictions = payload.get("predictions", payload.get("results"))
    if not isinstance(predictions, dict) or len(predictions) < 2:
        raise ValueError("Replay needs at least two predicted temporal frames")
    manifest = args.eap_root / "data/train.parquet"
    columns = [
        "sample_token",
        "split",
        "sequence_id",
        "rgb_shard_path",
        "rgb_member_path",
        "events_path",
        "rgb_exposure_start_timestamp_us",
        "rgb_exposure_end_timestamp_us",
        "K_event",
        "T_event_ego",
    ]
    rows = pq.read_table(
        manifest,
        columns=columns,
        filters=[("sample_token", "in", list(predictions))],
        use_threads=False,
    ).to_pylist()
    media = {str(row["sample_token"]): row for row in rows}
    frames = build_frame_plan(
        predictions,
        media,
        minimum_score=args.minimum_score,
        nms_iou=args.nms_iou,
    )
    return {
        "payload": payload,
        "media": media,
        "manifest": manifest,
        "predictions": predictions,
    }, frames


def _verify_candidate(path: Path) -> dict[str, Any]:
    from scripts.final_train40_transfer import verify_candidate

    _, binding = verify_candidate(path)
    return binding


def prepare(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate every immutable replay input without constructing CUDA state."""
    admitted, frames = _read_inputs(args)
    tracker = CausalMultiObjectTracker(
        minimum_iou=args.association_iou,
        ambiguity_margin=args.ambiguity_margin,
        confirm_hits=2,
        max_gap_us=args.max_gap_us,
    )
    prefix = replay_tracker_prefix(frames, tracker)
    final = frames[-1]
    preview = tracker.update(final["end_us"], final["detections"])
    mature = [
        item
        for item in preview.assignments
        if item.phase_admissible
        and final["end_us"] - item.track.first_timestamp_us >= args.minimum_track_history_us
    ]
    event_path = args.eap_root / admitted["media"][final["sample_token"]]["events_path"]
    if not event_path.is_file():
        raise FileNotFoundError(event_path)
    binding = _verify_candidate(args.candidate_root)
    phase_context_source = ROOT / "operational/detection_phase_service/phase_context.py"
    chunked_runtime_source = ROOT / "operational/detection_phase_service/chunked_runtime.py"
    report = {
        "schema": (
            "detection_phase_service_replay_phase_bounds_v1"
            if args.phase_bounds
            else "detection_phase_service_replay_v1"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PREPARED_NOT_INFERRED",
        "scope": "development_exposed_label_free_integration_no_accuracy_claim",
        "predictions_sha256": sha256_file(args.predictions),
        "replay_script_sha256": sha256_file(Path(__file__)),
        "phase_context": {
            "enabled": args.phase_bounds,
            "version": "batched_native_phase_context_v1" if args.phase_bounds else None,
            "source_sha256": sha256_file(phase_context_source) if args.phase_bounds else None,
            "graphs": False if args.phase_bounds else None,
        },
        "media_manifest_sha256": sha256_file(admitted["manifest"]),
        "candidate_binding": binding,
        "candidate_root": str(args.candidate_root.resolve()),
        "sequence_id": admitted["media"][final["sample_token"]]["sequence_id"],
        "frame_count": len(frames),
        "sensor_span_us": final["end_us"] - frames[0]["end_us"],
        "final_sample_token": final["sample_token"],
        "final_cutoff_us": final["end_us"],
        "final_detection_count": len(final["detections"]),
        "final_filter_accounting": final["filter_accounting"],
        "frame_filter_accounting": {
            frame["sample_token"]: frame["filter_accounting"] for frame in frames
        },
        "mature_unambiguous_track_count": len(mature),
        "mature_object_ids": [item.track.object_id for item in mature],
        "tracker_prefix": prefix,
        "event_source": {"path": str(event_path), "size_bytes": event_path.stat().st_size},
        "gpu_execution_requested": not args.prepare_only,
        "object_capacity": {
            "sealed_chunk_capacity": args.max_objects,
            "logical_capacity": args.logical_max_objects,
            "ingest_once_chunking": args.logical_max_objects > args.max_objects,
            "chunked_runtime_version": (
                "train40_ingest_once_chunked_v1"
                if args.logical_max_objects > args.max_objects
                else None
            ),
            "chunked_runtime_source_sha256": (
                sha256_file(chunked_runtime_source)
                if args.logical_max_objects > args.max_objects
                else None
            ),
        },
        "targets_read": False,
        "optimizer_updates": 0,
        "phase_bounds_semantics": "native phase q10/q90 or explicit null; never TTC interval",
    }
    return report, admitted, frames


def execute_gpu(
    args: argparse.Namespace,
    report: dict[str, Any],
    admitted: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load the sealed graph and perform the final real multi-object update."""
    if args.device != "cuda":
        raise ValueError("Actual MultiObjectTrain40Service replay requires --device cuda")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    execution_started = time.perf_counter()

    from e_jepa_ttc.data.eap import EAPEventReader
    from e_jepa_ttc.rgb_port.data import TarFrameReader
    from operational.detection_phase_service import ChunkedTrain40Runtime, DetectionPhaseService
    from operational.detection_phase_service.phase_context import BatchedPhaseContextRuntime
    from operational.train40_runtime.integral import IntegralEventHistory
    from operational.train40_runtime.multiobject import MultiObjectTrain40Service, RawRGBHistory
    from scripts.final_train40_transfer import load_models, verify_candidate

    paths, binding = verify_candidate(args.candidate_root)
    models, normalizers = load_models(paths, "cuda")
    max_objects = args.max_objects
    history = IntegralEventHistory(
        device="cuda",
        max_objects=max_objects,
        roi_batch_size=min(4, max_objects),
    )
    runtime_backend = MultiObjectTrain40Service(
        models,
        normalizers,
        history,
        device="cuda",
        graphs=False,
        parallel_branches=False,
        max_objects=max_objects,
        rgb_workers=args.rgb_workers,
        rgb_object_chunk_size=args.rgb_object_chunk_size,
        overlap_event_with_rgb_preparation=False,
        rgb_uint8=False,
    )
    if args.phase_bounds:
        runtime_backend.context = BatchedPhaseContextRuntime(
            models,
            normalizers,
            "cuda",
            graphs=False,
            max_objects=max_objects,
        )
    runtime = (
        ChunkedTrain40Runtime(
            runtime_backend,
            max_objects=args.logical_max_objects,
            chunk_size=max_objects,
        )
        if args.logical_max_objects > max_objects
        else runtime_backend
    )
    tracker = CausalMultiObjectTracker(
        minimum_iou=args.association_iou,
        ambiguity_margin=args.ambiguity_margin,
        confirm_hits=2,
        max_gap_us=args.max_gap_us,
    )
    prefix = replay_tracker_prefix(frames, tracker)
    final = frames[-1]
    query_us = int(final["end_us"])
    event_path = args.eap_root / admitted["media"][final["sample_token"]]["events_path"]
    with EAPEventReader(event_path) as reader:
        old = reader.read_window(query_us - 620_000, query_us - 20_000)
        packet = reader.read_window(query_us - 20_000, query_us)
    history.initialize(old, query_us - 620_000, query_us - 20_000)

    tokens = [frame["sample_token"] for frame in frames]
    frame_reader = TarFrameReader(args.eap_root, max_frame_cache_bytes=128 * 1024**2)
    try:
        images = {
            index: frame_reader.read_uint8(
                admitted["media"][token]["rgb_shard_path"],
                admitted["media"][token]["rgb_member_path"],
            )
            for index, token in enumerate(tokens)
        }
    finally:
        frame_reader.close()
    timestamps = {
        index: int(admitted["media"][token]["rgb_exposure_end_timestamp_us"])
        for index, token in enumerate(tokens)
    }
    groups = causal_rgb_groups(timestamps, query_us)
    rgb = RawRGBHistory(query_us, images, timestamps, groups)
    service = DetectionPhaseService(
        runtime,
        tracker=tracker,
        minimum_track_history_us=args.minimum_track_history_us,
    )
    try:
        result = service.update(
            packet,
            query_us - 20_000,
            query_us,
            final["detections"],
            rgb,
        )
        torch.cuda.synchronize()
    finally:
        service.close()
    report.update(
        status="EXECUTED_DEVELOPMENT_EXPOSED",
        candidate_binding=binding,
        tracker_prefix=prefix,
        result=result,
        event_initialization_count=len(old["t"]),
        event_packet_count=len(packet["t"]),
        phase_bounds=(
            "null_explicit_in_current_batched_runtime"
            if all(
                item["phase_q10"] is None and item["phase_q90"] is None
                for item in result["detections"]
            )
            else "native_phase_bounds_preserved"
        ),
        cuda_resources={
            "device": torch.cuda.get_device_name(),
            "device_total_bytes": torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "execution_wall_seconds_including_model_load_and_media_read": (
                time.perf_counter() - execution_started
            ),
            "max_objects": max_objects,
            "logical_max_objects": args.logical_max_objects,
            "rgb_object_chunk_size": args.rgb_object_chunk_size,
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--phase-bounds",
        action="store_true",
        help="Opt in to native frozen PHASE q10/q90 transport (graphs disabled)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--minimum-score", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--association-iou", type=float, default=0.3)
    parser.add_argument("--ambiguity-margin", type=float, default=0.1)
    parser.add_argument("--max-gap-us", type=int, default=150_000)
    parser.add_argument("--minimum-track-history-us", type=int, default=600_000)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument(
        "--logical-max-objects",
        type=int,
        default=None,
        help="Total admitted objects; values above --max-objects use ingest-once chunks",
    )
    parser.add_argument("--rgb-object-chunk-size", type=int, default=1)
    parser.add_argument("--rgb-workers", type=int, default=2)
    args = parser.parse_args()
    if args.logical_max_objects is None:
        args.logical_max_objects = args.max_objects
    if args.output.exists():
        raise FileExistsError("Use a new output path")
    if not 1 <= args.max_objects <= 20:
        parser.error("--max-objects must be in 1..20")
    if not 1 <= args.rgb_object_chunk_size <= args.max_objects:
        parser.error("RGB chunk size must be in 1..max-objects")
    if args.logical_max_objects < args.max_objects:
        parser.error("--logical-max-objects must be at least --max-objects")
    if args.prepare_only and args.device != "cpu":
        parser.error("Preparation is CPU-only; use --device cpu")
    report, admitted, frames = prepare(args)
    if not args.prepare_only:
        report = execute_gpu(args, report, admitted, frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
