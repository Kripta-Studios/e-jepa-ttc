"""Predict signed TTC/phase bounds from detector-only causal eAP observations.

No annotations, Garl IDs, optimizer updates, or test targets are used. Association
can abstain. A shared crop preserves pixel scale; each past detection remains an
independent measured prediction. Heavy imports happen only after CLI validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Stream a provenance hash without retaining file contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box_iou(first: list[float], second: list[float]) -> float:
    """Compute image-plane IoU for positive xyxy boxes."""
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(area_first + area_second - intersection, 1e-12)


def suppress_duplicates(detections: list[dict], threshold: float) -> tuple[list[dict], int]:
    """Class-aware NMS only for choosing TTC regions, without editing detector results."""
    retained = []
    for item in sorted(detections, key=lambda d: (-d["score"], d["prediction_index"])):
        if any(
            previous["class_name"] == item["class_name"]
            and box_iou(previous["roi_xyxy"], item["roi_xyxy"]) > threshold
            for previous in retained
        ):
            continue
        retained.append(item)
    return retained, len(detections) - len(retained)


def causal_tracks(
    frames: list[dict], *, iou_threshold: float, ambiguity_margin: float, max_gap_us: int
) -> list[dict]:
    """Associate chronologically with conservative one-to-one class/IoU matching.

    An ambiguous detection starts a fresh track. No future observation repairs
    previous assignments. The selected current detection is always the highest
    score after declared NMS; there is no fallback to an easier object.
    """
    previous: list[dict] = []
    previous_time: int | None = None
    serial = 0
    audit = []
    for frame in frames:
        now = int(frame["timestamp_us"])
        if previous_time is not None and now <= previous_time:
            raise ValueError("Association timestamps must increase strictly")
        if previous_time is not None and now - previous_time > max_gap_us:
            previous = []
        current = frame["detections"]
        edges = []
        diagnostics: dict[int, dict] = {}
        for index, item in enumerate(current):
            candidates = sorted(
                [
                    (box_iou(item["roi_xyxy"], old["roi_xyxy"]), pi)
                    for pi, old in enumerate(previous)
                    if old["class_name"] == item["class_name"]
                ],
                reverse=True,
            )
            best = candidates[0][0] if candidates else 0.0
            second = candidates[1][0] if len(candidates) > 1 else 0.0
            ambiguous = len(candidates) > 1 and best - second < ambiguity_margin
            diagnostics[index] = {"best_iou": best, "second_iou": second, "ambiguous": ambiguous}
            if candidates and best >= iou_threshold and not ambiguous:
                edges.append((best, index, candidates[0][1]))
        assignments: dict[int, int] = {}
        used_previous = set()
        for _, index, previous_index in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
            if previous_index not in used_previous:
                assignments[index] = previous_index
                used_previous.add(previous_index)
        for index, item in enumerate(current):
            if index in assignments:
                old = previous[assignments[index]]
                item["track_id"] = old["track_id"]
                item["history"] = old["history"] + [
                    {
                        "sample_token": frame["sample_token"],
                        "timestamp_us": now,
                        "roi_xyxy": item["roi_xyxy"],
                        "prediction_index": item["prediction_index"],
                    }
                ]
                reason = "matched"
            else:
                serial += 1
                item["track_id"] = f"detected_roi_{serial:04d}"
                item["history"] = [
                    {
                        "sample_token": frame["sample_token"],
                        "timestamp_us": now,
                        "roi_xyxy": item["roi_xyxy"],
                        "prediction_index": item["prediction_index"],
                    }
                ]
                reason = "ambiguous" if diagnostics[index]["ambiguous"] else "new_or_unmatched"
            audit.append(
                {
                    "sample_token": frame["sample_token"],
                    "prediction_index": item["prediction_index"],
                    "track_id": item["track_id"],
                    "reason": reason,
                    **diagnostics[index],
                }
            )
        previous, previous_time = current, now
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare real sensor history, abstain on insufficient support, then run PHASE."""
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]
    import numpy as np
    import pyarrow.parquet as pq

    from e_jepa_ttc.detection.geometry_contract import original_bundle_corners

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    forbidden = {"ground_truth", "labels", "annotations", "ttc", "target"}
    if forbidden & set(payload):
        raise ValueError("Use a predictions-only JSON; mixed evaluation/GT files are refused")
    predictions = payload.get("predictions", payload.get("results"))
    if not isinstance(predictions, dict) or args.sample_token not in predictions:
        raise ValueError("The predictions-only input lacks the selected query token")
    sequence = args.sample_token.split(":", 1)[0]
    if any(not token.startswith(sequence + ":") for token in predictions):
        raise ValueError("One sequence per bridge run is required")
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
    manifest = args.eap_root / "data/train.parquet"
    rows = pq.read_table(
        manifest,
        columns=columns,
        filters=[("sequence_id", "=", sequence), ("sample_token", "in", list(predictions))],
        use_threads=False,
    ).to_pylist()
    media = {row["sample_token"]: row for row in rows}
    if any(token not in media for token in predictions):
        raise ValueError("Every predicted token must exist in the public train manifest")
    current = media[args.sample_token]
    query_us = int(current["rgb_exposure_end_timestamp_us"])
    anchor_us = int(current["rgb_exposure_start_timestamp_us"])
    frames = []
    rejected = []
    for token in sorted(predictions, key=lambda key: media[key]["rgb_exposure_start_timestamp_us"]):
        row = media[token]
        start, end = (
            int(row[key])
            for key in ["rgb_exposure_start_timestamp_us", "rgb_exposure_end_timestamp_us"]
        )
        if row["split"] != "train" or end < start:
            raise ValueError("Train-only ordered exposures are required")
        if start > anchor_us or end > query_us or query_us - start > 650_000:
            continue
        detections = []
        for index, item in enumerate(predictions[token]):
            if forbidden & set(item):
                raise ValueError("Detection entries must not carry targets")
            score = float(item["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Finite confidence in [0,1] is required")
            if score < args.minimum_score:
                continue
            length, width, height = item["size_lwh_m"]
            box = [*item["translation_m"], length, height, width, item["yaw_rad"]]
            try:
                corners = original_bundle_corners(box)
                camera = (
                    np.column_stack((corners, np.ones(8)))
                    @ np.asarray(row["T_event_ego"], dtype=float).T
                )[:, :3]
                if np.any(camera[:, 2] <= 1e-4):
                    raise ValueError("Box has corners at or behind camera plane")
                projected = camera @ np.asarray(row["K_event"], dtype=float).T
                uv = projected[:, :2] / projected[:, 2:3]
                roi = np.clip(np.r_[uv.min(0), uv.max(0)], 0, [1279, 719, 1279, 719])
                if np.any(roi[2:] - roi[:2] < 8):
                    raise ValueError("Projected ROI smaller than 8 pixels")
            except ValueError as error:
                rejected.append(
                    {"sample_token": token, "prediction_index": index, "reason": str(error)}
                )
                continue
            detections.append(
                {
                    "score": score,
                    "class_name": str(item["class_name"]),
                    "prediction_index": index,
                    "roi_xyxy": roi.tolist(),
                }
            )
        detections, suppressed = suppress_duplicates(detections, args.nms_iou)
        frames.append(
            {
                "sample_token": token,
                "timestamp_us": start,
                "exposure_end_us": end,
                "detections": detections,
                "duplicate_predictions_suppressed": suppressed,
            }
        )
    association = causal_tracks(
        frames,
        iou_threshold=args.association_iou,
        ambiguity_margin=args.ambiguity_margin,
        max_gap_us=150_000,
    )
    report: dict[str, Any] = {
        "schema": "predicted_eap_roi_phase_bridge_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "UNAVAILABLE",
        "sample_token": args.sample_token,
        "sensor_anchor_us": anchor_us,
        "decision_sensor_cutoff_us": query_us,
        "physical_arrival_timestamps_measured": False,
        "prediction_file_sha256": file_sha256(args.predictions),
        "script_sha256": file_sha256(Path(__file__)),
        "media_manifest_sha256": file_sha256(manifest),
        "detector_metadata": {
            key: payload.get(key)
            for key in ("checkpoint_sha256", "preregistration_sha256", "inference_source_sha256")
        },
        "association_policy": {
            "class_aware_nms_iou": args.nms_iou,
            "minimum_score": args.minimum_score,
            "greedy_class_iou_minimum": args.association_iou,
            "minimum_iou_margin": args.ambiguity_margin,
            "max_gap_us": 150_000,
            "current_selection": "highest confidence; no easier-object fallback",
        },
        "association": association,
        "rejected_rois": rejected,
        "outputs": {"ttc": None, "point_phase": None, "q10": None, "q90": None},
        "output_space": {
            "ttc": "signed seconds",
            "point_phase": "phase",
            "q10": "phase",
            "q90": "phase",
        },
        "phase_boundaries_are_calibrated_intervals": False,
        "ground_truth_read": False,
        "garl_ids_used": False,
        "optimizer_updates": 0,
        "scope": "development_exposed interface demo; no TTC accuracy or official metric",
    }
    if (
        not frames
        or frames[-1]["sample_token"] != args.sample_token
        or not frames[-1]["detections"]
    ):
        return {**report, "reason": "no_current_usable_detection"}
    selected = frames[-1]["detections"][0]
    history = selected["history"]
    report.update(
        track_id=selected["track_id"],
        class_name=selected["class_name"],
        detection_score=selected["score"],
        detected_history=history,
    )
    if len(history) < 2:
        return {**report, "reason": "insufficient_unambiguous_causal_detection_history"}

    import torch

    from e_jepa_ttc.data.eap import EAPEventReader
    from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
    from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
    from e_jepa_ttc.rgb_port.data import TarFrameReader, crop_uint8
    from operational.evttc_competition.runtime import CompetitionRuntime, PreparedObservation
    from scripts.final_train40_transfer import verify_candidate

    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    boxes = [item["roi_xyxy"] for item in history]
    requested = common_square_from_boxes(boxes, range(len(boxes)), margin_fraction=0.25)
    square = (
        max(0, int(requested[0])),
        max(0, int(requested[1])),
        min(1280, int(requested[2])),
        min(720, int(requested[3])),
    )
    report.update(
        common_roi_xyxy=square,
        requested_common_roi_xyxy=requested,
        roi_policy="same clipped union ROI for every real RGB/event observation",
        event_x_offset_pixels=args.event_x_offset,
    )
    clocks = [int(item["timestamp_us"]) for item in history]
    groups = [list(range(i - 2, i + 1)) for i in range(2, len(history))]
    if len(history) == 2:
        groups = [[0, 1]]
    frame_reader = TarFrameReader(args.eap_root, max_frame_cache_bytes=32 * 1024**2)
    rgb_observations = []
    event_specs = []
    observation_receipts = []
    try:
        for group in groups:
            times = [clocks[index] for index in group]
            pixels = np.stack(
                [
                    crop_uint8(
                        frame_reader.read_uint8(
                            media[history[index]["sample_token"]]["rgb_shard_path"],
                            media[history[index]["sample_token"]]["rgb_member_path"],
                        ),
                        square,
                    )
                    for index in group
                ]
            )
            rgb_observations.append(
                PreparedObservation(
                    "rgb",
                    pixels.astype(np.float32) / 255.0,
                    tuple(times),
                    times[0],
                    times[-1],
                    query_us,
                )
            )
            boundaries = [times[0] - (times[1] - times[0]), *times]
            if query_us - boundaries[0] <= 650_000:
                event_specs.append((times, list(zip(boundaries[:-1], boundaries[1:], strict=True))))
            observation_receipts.append(
                {
                    "frame_tokens": [history[index]["sample_token"] for index in group],
                    "sensor_endpoints_us": times,
                    "available_at_sensor_cutoff_us": query_us,
                    "delta_t_s": (np.diff(times) / 1e6).tolist(),
                }
            )
    finally:
        frame_reader.close()
    if not event_specs:
        return {**report, "reason": "no_complete_event_observation_inside_650ms"}
    event_path = args.eap_root / current["events_path"]
    low = min(interval[0] for _, windows in event_specs for interval in windows)
    high = max(interval[1] for _, windows in event_specs for interval in windows)
    chunks = []
    retained = 0
    with EAPEventReader(event_path) as reader:
        if low < reader.t_start_us or high > reader.t_end_us:
            return {**report, "reason": "event_source_lacks_complete_causal_window"}
        for raw in reader.iter_window_chunks(low, high, chunk_events=100_000):
            mapped = map_roi(raw, square, 128, args.event_x_offset)
            retained += sum(value.nbytes for value in mapped.values())
            if retained > 128 * 1024**2:
                raise MemoryError("Mapped event ROI exceeds the 128 MiB bridge budget")
            chunks.append(mapped)
    mapped = {
        key: np.concatenate([chunk[key] for chunk in chunks])
        if chunks
        else np.empty(0, dtype=np.int64)
        for key in ("x", "y", "t", "p")
    }
    event_observations = []
    for times, windows in event_specs:
        pixels = torch.stack(
            [encode_mapped(mapped, start, end, 128, sequence) for start, end in windows]
        )
        event_observations.append(
            PreparedObservation(
                "event", pixels.numpy(), tuple(times), windows[0][0], times[-1], query_us
            )
        )
    report.update(
        observations=observation_receipts,
        event_windows_us=[windows for _, windows in event_specs],
        event_roi_count=len(mapped["t"]),
        event_retained_bytes=retained,
        event_window_boundary="half-open [start,end); exact media/event clock",
        event_source={
            "relative_path": current["events_path"],
            "size_bytes": event_path.stat().st_size,
        },
    )
    del mapped, chunks
    if args.prepare_only:
        return {**report, "status": "PREPARED_NOT_INFERRED", "reason": "prepare_only"}
    _, binding = verify_candidate(args.candidate_root)
    runtime = CompetitionRuntime.from_candidate(args.candidate_root, device=args.device)
    captured = {}

    def capture(_model: Any, _inputs: Any, output: dict) -> None:
        for name in ("ttc", "point_phase", "q10", "q90", "raw_location"):
            captured[name] = float(output[name][0].detach().cpu())

    hook = runtime.models["PHASE_ONLY_EVENT_S7_TRAIN40"].register_forward_hook(capture)
    try:
        prediction = runtime.predict(event_observations, rgb_observations, query_us)
    finally:
        hook.remove()
    from dataclasses import asdict

    report.update(phase_model_binding=binding, runtime=asdict(prediction))
    report["prediction_finished_at_utc"] = datetime.now(UTC).isoformat()
    report["phase_source_sha256"] = {
        path: file_sha256(root / path)
        for path in (
            "operational/evttc_competition/runtime.py",
            "operational/final_refit/test12.py",
            "operational/final_refit/training.py",
            "operational/final_refit/features.py",
            "src/e_jepa_ttc/simplex_t/phase.py",
        )
    }
    if prediction.status != "ok" or not captured:
        return {
            **report,
            "reason": prediction.model_reason,
            "event_only_ttc_if_available": prediction.event_ttc,
        }
    if not all(math.isfinite(value) for value in captured.values()):
        raise ValueError("Nonfinite final PHASE output")
    if captured["q10"] > captured["q90"]:
        raise ValueError("Final phase boundaries are reversed")
    if not math.isclose(captured["ttc"], float(prediction.model_ttc), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Captured head TTC differs from the native runtime output")
    return {
        **report,
        "status": "PREDICTED_DEVELOPMENT_EXPOSED",
        "reason": "fused_event_rgb",
        "outputs": captured,
        "phase_interval_crosses_zero": captured["q10"] < 0 < captured["q90"],
    }


def main() -> int:
    """Run one preselected query with declared causal association thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-token", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--minimum-score", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--association-iou", type=float, default=0.3)
    parser.add_argument("--ambiguity-margin", type=float, default=0.1)
    parser.add_argument("--event-x-offset", type=float, default=5.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1 or any(
        not 0 <= value <= 1
        for value in (args.minimum_score, args.nms_iou, args.association_iou, args.ambiguity_margin)
    ):
        parser.error("Positive CPU threads and thresholds in [0,1] are required")
    if args.output.exists():
        raise FileExistsError("Use a new output path; previous bridge evidence is preserved")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "reason": report["reason"], "outputs": report["outputs"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
