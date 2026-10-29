"""Prepare, run and score the five-query detected-PHASE development panel.

Preparation and inference never read TTC labels. Scoring first seals the prediction
bytes, then opens the exact public-TRAIN annotation keys. This is an exposed
TRAIN40 diagnostic, not an independent-generalization result.
"""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import weakref
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from e_jepa_ttc.detection.causal_tracker import (
    CausalMultiObjectTracker,
    DetectionObservation,
    Track,
    TrackAssignment,
    TrackerUpdate,
    TrackState,
)
from scripts.build_ttc_development_panel import JOIN_KEYS, evaluator_assignment
from scripts.replay_detection_phase_service import project_detection, suppress_duplicates

CONDITIONS = ("R0", "R1", "R2", "R3")
FORBIDDEN_INFERENCE_KEYS = {
    "ttc",
    "ttc_true",
    "ground_truth",
    "annotations",
    "target",
    "reference_roi",
    "reference_id",
}


class OracleReferenceTracker:
    """Diagnostic-only tracker that preserves one evaluator reference identity."""

    def __init__(self, reference_id: str) -> None:
        if not reference_id:
            raise ValueError("Reference identity must be nonempty")
        self.reference_id = reference_id
        self._track: Track | None = None
        self._last_timestamp_us: int | None = None

    def update(self, timestamp_us: int, detections: list[DetectionObservation]) -> TrackerUpdate:
        """Assign the sole evaluator-matched detection to the reference generation."""
        if self._last_timestamp_us is not None and timestamp_us <= self._last_timestamp_us:
            raise ValueError("Oracle tracker timestamps must increase strictly")
        if len(detections) > 1:
            raise ValueError("Oracle condition admits at most one matched target detection")
        self._last_timestamp_us = timestamp_us
        assignments = []
        if detections:
            detection = detections[0]
            first = timestamp_us if self._track is None else self._track.first_timestamp_us
            hits = 1 if self._track is None else self._track.hits + 1
            self._track = Track(
                track_id=self.reference_id,
                generation=1,
                class_name=detection.class_name,
                state=TrackState.CONFIRMED,
                hits=hits,
                missed=0,
                first_timestamp_us=first,
                last_timestamp_us=timestamp_us,
                event_square=detection.event_square,
                rgb_square=detection.rgb_square,
            )
            assignments.append(
                TrackAssignment(
                    detection=detection,
                    track=self._track,
                    association_status="new" if hits == 1 else "matched",
                    association_cost=0.0,
                    ambiguity_reason="",
                )
            )
        return TrackerUpdate(
            timestamp_us=timestamp_us,
            assignments=tuple(assignments),
            tracks=(() if self._track is None else (self._track,)),
        )


def register_fresh_runtime(runtime: Any, references: list[weakref.ReferenceType[Any]]) -> None:
    """Reject cache-bearing runtime reuse without retaining prior GPU runtimes."""
    references[:] = [reference for reference in references if reference() is not None]
    if any(reference() is runtime for reference in references):
        raise ValueError("Every query/condition requires a fresh runtime instance")
    try:
        references.append(weakref.ref(runtime))
    except TypeError as error:
        raise TypeError("Runtime instances must support weak references") from error


def sha256_file(path: Path) -> str:
    """Hash one artifact incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash one finite JSON value canonically."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def snapshot_json(path: Path) -> tuple[bytes, str, Any]:
    """Read, hash and parse exactly one immutable byte snapshot."""
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest(), json.loads(data)


def array_sha256(value: object) -> str:
    """Hash one calibration array as float64 C-order bytes."""
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes(order="C")).hexdigest()


def _finite_box(value: Sequence[float]) -> list[float]:
    box = [float(item) for item in value]
    if len(box) != 4 or not all(math.isfinite(item) for item in box):
        raise ValueError("ROI must contain four finite values")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("ROI must have positive extent")
    return box


def perturb_roi(value: Sequence[float], width: int = 1280, height: int = 720) -> list[float]:
    """Apply the preregistered +5% translation, shifted inward at image borders."""
    x1, y1, x2, y2 = _finite_box(value)
    box_width, box_height = x2 - x1, y2 - y1
    x1 += 0.05 * box_width
    x2 += 0.05 * box_width
    y1 += 0.05 * box_height
    y2 += 0.05 * box_height
    if x2 > width:
        x1, x2 = x1 - (x2 - width), float(width)
    if y2 > height:
        y1, y2 = y1 - (y2 - height), float(height)
    if x1 < 0:
        x2, x1 = x2 - x1, 0.0
    if y1 < 0:
        y2, y1 = y2 - y1, 0.0
    return _finite_box((x1, y1, x2, y2))


def _validate_prediction_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Validate one already snapshotted detector payload."""
    if FORBIDDEN_INFERENCE_KEYS & set(payload):
        raise ValueError("Detector artifact contains forbidden supervision")
    predictions = payload.get("predictions", payload.get("results"))
    checkpoint_sha = payload.get("checkpoint_sha256")
    inference_source_sha = payload.get("inference_source_sha256")
    if (
        not isinstance(predictions, dict)
        or not isinstance(checkpoint_sha, str)
        or len(checkpoint_sha) != 64
        or not isinstance(inference_source_sha, str)
        or len(inference_source_sha) != 64
        or payload.get("ground_truth_in_output") is not False
    ):
        raise ValueError("Detector artifact needs label-free predictions and checkpoint SHA-256")
    for token, rows in predictions.items():
        if not isinstance(rows, list):
            raise ValueError(f"Predictions must be lists: {token}")
        for row in rows:
            if not isinstance(row, dict) or FORBIDDEN_INFERENCE_KEYS & set(row):
                raise ValueError(f"Detector row contains supervision: {token}")
    return payload, predictions


def _prediction_payload(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    return _validate_prediction_payload(json.loads(path.read_bytes()))


def _exact_index(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str], name: str
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def prepare_phase(
    panel_path: Path,
    predictions_path: Path,
    eap_root: Path,
    garl_root: Path,
) -> dict[str, Any]:
    """Build label-free R3 inputs and explicitly separated oracle inputs."""
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    queries = panel.get("queries")
    if panel.get("schema") != "ttc_development_five_query_panel_v1" or len(queries or []) != 5:
        raise ValueError("Expected the corrected five-query panel")
    detector, predictions = _prediction_payload(predictions_path)
    data_receipts = detector.get("data_receipts", {})
    if not isinstance(data_receipts, dict):
        raise ValueError("Detector data receipts must be a mapping")
    required_tokens = sorted(
        {token for query in queries for token in query["prefix"]["frame_tokens"]}
    )
    media_path = eap_root / "data/train.parquet"
    media_rows = pq.read_table(
        media_path,
        columns=[
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
        ],
        filters=[("sample_token", "in", required_tokens)],
        use_threads=False,
    ).to_pylist()
    media = _exact_index(media_rows, ("sample_token",), "media token")
    if set(token[0] for token in media) != set(required_tokens):
        raise ValueError("Panel prefix token is absent from public TRAIN media")
    if any(row["split"] != "train" for row in media.values()):
        raise ValueError("Panel media escaped public TRAIN")

    garl_path = garl_root / "data/train.parquet"
    query_tokens = [query["garl_query_token"] for query in queries]
    garl_rows = pq.read_table(
        garl_path,
        columns=[*JOIN_KEYS, "rgb_member_paths", "boxes_xyxy"],
        filters=[("sample_token", "in", query_tokens)],
        use_threads=False,
    ).to_pylist()
    garl = _exact_index(garl_rows, JOIN_KEYS, "Garl query")

    missing = sorted(set(required_tokens) - set(predictions))
    extra = sorted(set(predictions) - set(required_tokens))
    prepared_queries = []
    for query in queries:
        if int(query["prefix"]["span_us"]) < 600_000:
            raise ValueError("Every query needs at least 600 ms of causal history")
        join_key = tuple(query["target_annotation_join_key"])
        source = garl.get(join_key)
        if source is None:
            raise ValueError("Panel query lacks its exact Garl input row")
        boxes_by_member = {
            str(member): _finite_box(box)
            for member, box in zip(source["rgb_member_paths"], source["boxes_xyxy"], strict=True)
        }
        prefix = []
        for token, member in zip(
            query["prefix"]["frame_tokens"], query["prefix"]["rgb_members"], strict=True
        ):
            row = media[(token,)]
            if str(row["rgb_member_path"]) != str(member):
                raise ValueError("Panel token/member identity differs from public TRAIN media")
            prefix.append(
                {
                    "sample_token": token,
                    "sequence_id": str(row["sequence_id"]),
                    "rgb_member_path": str(row["rgb_member_path"]),
                    "rgb_shard_path": str(row["rgb_shard_path"]),
                    "events_path": str(row["events_path"]),
                    "exposure_start_us": int(row["rgb_exposure_start_timestamp_us"]),
                    "exposure_end_us": int(row["rgb_exposure_end_timestamp_us"]),
                    "K_event": np.asarray(row["K_event"], dtype=np.float64).tolist(),
                    "T_event_ego": np.asarray(row["T_event_ego"], dtype=np.float64).tolist(),
                    "K_event_sha256_float64_c_order": array_sha256(row["K_event"]),
                    "T_event_ego_sha256_float64_c_order": array_sha256(row["T_event_ego"]),
                    "rgb_member_sha256": (data_receipts.get(token, {}).get("rgb_member_sha256")),
                    "detector_predictions_present": token in predictions,
                }
            )
        reference = [
            {
                "sample_token": frame["sample_token"],
                "reference_id": query["garl_public_track_id_evaluation_only"],
                "roi_xyxy": boxes_by_member.get(frame["rgb_member_path"]),
            }
            for frame in prefix
        ]
        prepared_queries.append(
            {
                "query_id": query["garl_query_token"],
                "sequence_id": query["sequence_id"],
                "target_frame_token": query["target_frame_token"],
                "target_annotation_join_key_hash": canonical_sha256(join_key),
                "label_age_at_prediction_us": query["timing"]["label_age_at_prediction_us"],
                "sensor_reference_us": query["timing"]["sensor_reference_us"],
                "prediction_cutoff_us": query["timing"]["prediction_available_cutoff_us"],
                "ttc_target_shift_applied_us": 0,
                "r3_label_free": {"prefix": prefix},
                "oracle": {
                    "R0": reference,
                    "R1": [
                        {**item, "roi_xyxy": perturb_roi(item["roi_xyxy"])}
                        if item["roi_xyxy"] is not None
                        else item
                        for item in reference
                    ],
                    "R2": {
                        "reference_identity": query["garl_public_track_id_evaluation_only"],
                        "reference_rois": reference,
                        "identity_assignment": "evaluator_only_after_detector_output",
                    },
                },
                "readiness": {
                    condition: query["conditions"][
                        {
                            "R0": "R0_reference_roi_reference_id",
                            "R1": "R1_perturbed_roi_reference_id",
                            "R2": "R2_detected_roi_reference_id",
                            "R3": "R3_fully_predicted",
                        }[condition]
                    ]
                    for condition in CONDITIONS
                },
            }
        )
    missing_media_hashes = sorted(
        token
        for token in required_tokens
        if not isinstance(data_receipts.get(token, {}).get("rgb_member_sha256"), str)
    )
    return {
        "schema": "detected_phase_panel_preparation_v1",
        "status": (
            "READY_FOR_RUN"
            if not missing and not extra and not missing_media_hashes
            else "BLOCKED_PREDICTION_COVERAGE"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "TRAIN40_EXPOSED_DIAGNOSTIC_ONLY",
        "panel_sha256": sha256_file(panel_path),
        "harness_sha256": sha256_file(Path(__file__)),
        "phase_service_sha256": sha256_file(
            ROOT / "operational/detection_phase_service/service.py"
        ),
        "replay_geometry_source_sha256": sha256_file(
            ROOT / "scripts/replay_detection_phase_service.py"
        ),
        "causal_tracker_sha256": sha256_file(ROOT / "src/e_jepa_ttc/detection/causal_tracker.py"),
        "geometry_contract_sha256": sha256_file(
            ROOT / "src/e_jepa_ttc/detection/geometry_contract.py"
        ),
        "multiobject_runtime_sha256": sha256_file(
            ROOT / "operational/train40_runtime/multiobject.py"
        ),
        "predictions_sha256": sha256_file(predictions_path),
        "detector_checkpoint_sha256": detector["checkpoint_sha256"],
        "detector_inference_source_sha256": detector.get("inference_source_sha256"),
        "media_manifest_sha256": sha256_file(media_path),
        "required_prefix_tokens": required_tokens,
        "missing_prediction_tokens": missing,
        "extraneous_prediction_tokens": extra,
        "missing_rgb_member_hash_tokens": missing_media_hashes,
        "targets_read": False,
        "gt_roi_or_identity_fed_to_R3": False,
        "queries": prepared_queries,
    }


def run_fresh_service(
    prepared: Mapping[str, Any],
    predictions: Mapping[str, list[dict[str, Any]]],
    runtime_factory: Callable[[str, Mapping[str, Any]], Any],
    frame_loader: Callable[
        [str, Mapping[str, Any], Mapping[str, Any]],
        tuple[Mapping[str, np.ndarray], Any] | tuple[Mapping[str, np.ndarray], Any, int, int],
    ],
) -> dict[str, Any]:
    """Execute every query/condition with a fresh existing DetectionPhaseService."""
    from operational.detection_phase_service.service import DetectionPhaseService

    if prepared["status"] != "READY_FOR_RUN":
        raise ValueError("Exact detector prefix coverage is required before run")
    results: dict[str, dict[str, Any]] = {condition: {} for condition in CONDITIONS}
    runtime_refs: list[weakref.ReferenceType[Any]] = []
    for condition in CONDITIONS:
        for query in prepared["queries"]:
            execution_query = (
                {
                    "query_id": query["query_id"],
                    "sequence_id": query["sequence_id"],
                    "r3_label_free": query["r3_label_free"],
                }
                if condition == "R3"
                else query
            )
            runtime = runtime_factory(condition, execution_query)
            register_fresh_runtime(runtime, runtime_refs)
            reference_id = query["oracle"]["R0"][-1]["reference_id"]
            tracker = (
                CausalMultiObjectTracker(
                    minimum_iou=0.3,
                    ambiguity_margin=0.05,
                    confirm_hits=2,
                    lost_ttl_frames=2,
                    max_gap_us=150_000,
                )
                if condition == "R3"
                else OracleReferenceTracker(reference_id)
            )
            service = DetectionPhaseService(
                runtime, tracker=cast(Any, tracker), minimum_track_history_us=600_000
            )
            packets = 0
            last = None
            try:
                for frame in query["r3_label_free"]["prefix"]:
                    loaded = frame_loader(condition, execution_query, frame)
                    if len(loaded) == 2:
                        packet, rgb_history = loaded
                        packet_start_us = int(frame["exposure_start_us"])
                        packet_end_us = int(frame["exposure_end_us"])
                    elif len(loaded) == 4:
                        packet, rgb_history, packet_start_us, packet_end_us = loaded
                    else:
                        raise ValueError(
                            "Frame loader must return media or media plus packet bounds"
                        )
                    detections = build_condition_detections(
                        condition, execution_query, frame, predictions[frame["sample_token"]]
                    )
                    last = service.update(
                        packet,
                        int(packet_start_us),
                        int(packet_end_us),
                        detections,
                        rgb_history,
                    )
                    if (
                        last.get("schema") != "detection_phase_multiobject_v1"
                        or last.get("packet_ingested_count") != 1
                        or last.get("targets_read") is not False
                    ):
                        raise ValueError(
                            "Service must attest schema, no targets, and one packet ingestion"
                        )
                    packets += 1
            finally:
                service.close()
            if packets != len(query["r3_label_free"]["prefix"]):
                raise AssertionError("Each prefix packet must be ingested exactly once")
            if last is not None:
                detections = [dict(item) for item in last.get("detections", [])]
                oracle_key = "R1" if condition == "R1" else "R0"
                target_roi = query["oracle"][oracle_key][-1]["roi_xyxy"]
                if target_roi is not None:
                    assignment = evaluator_assignment(
                        [item["event_square"] for item in detections], [target_roi]
                    )
                    matched_indices = {pair["predicted_index"] for pair in assignment["pairs"]}
                    for index, item in enumerate(detections):
                        item["evaluation_reference_match"] = index in matched_indices
                    reference_id = query["oracle"]["R0"][-1]["reference_id"]
                    matched = [item for item in detections if item["evaluation_reference_match"]]
                    if condition in {"R0", "R1", "R2"} and any(
                        item.get("track_id") != reference_id for item in matched
                    ):
                        raise ValueError(
                            f"{condition} must use the evaluator-assigned reference identity"
                        )
                    if condition == "R3" and any(
                        item.get("track_id") == reference_id for item in matched
                    ):
                        raise ValueError("R3 must use generated identity, never reference identity")
                    last = {**last, "detections": detections}
            results[condition][query["query_id"]] = {
                "packet_ingested_count": packets,
                "result": last,
            }
            del service, runtime
    return {
        "schema": "detected_phase_panel_predictions_v1",
        "status": "EXECUTED_TRAIN40_EXPOSED_DIAGNOSTIC",
        "prepared_sha256": canonical_sha256(prepared),
        "targets_read": False,
        "conditions_are_fresh_and_independent": True,
        "phase_bounds_semantics": "native q10/q90; never a continuous TTC interval",
        "results": results,
    }


def build_condition_detections(
    condition: str,
    query: Mapping[str, Any],
    frame: Mapping[str, Any],
    detector_rows: list[dict[str, Any]],
) -> Sequence[DetectionObservation]:
    """Create oracle R0/R1 or label-free detector R2/R3 observations."""
    token = str(frame["sample_token"])
    if condition in {"R0", "R1"}:
        reference = next(
            item for item in query["oracle"][condition] if item["sample_token"] == token
        )
        roi = reference["roi_xyxy"]
        if roi is None:
            return []
        box = _finite_box(roi)
        square = (box[0], box[1], box[2], box[3])
        return [
            DetectionObservation(
                detection_id=f"{condition}:{token}:{reference['reference_id']}",
                class_name="reference_object_oracle",
                score=1.0,
                event_square=square,
                rgb_square=square,
            )
        ]
    if condition not in {"R2", "R3"}:
        raise ValueError(f"Unknown condition: {condition}")
    intrinsic = np.asarray(frame["K_event"], dtype=np.float64)
    transform = np.asarray(frame["T_event_ego"], dtype=np.float64)
    detections = []
    for index, row in enumerate(detector_rows):
        observation = project_detection(
            row,
            token=token,
            index=index,
            intrinsic=intrinsic,
            event_from_ego=transform,
            minimum_score=0.1,
        )
        if observation is not None:
            detections.append(observation)
    retained = suppress_duplicates(detections, 0.5)
    if condition == "R3":
        if "oracle" in query:
            raise ValueError("R3 execution view must not contain oracle inputs")
        return retained
    reference = next(
        item for item in query["oracle"]["R2"]["reference_rois"] if item["sample_token"] == token
    )
    if reference["roi_xyxy"] is None:
        return []
    assignment = evaluator_assignment(
        [item.event_square for item in retained], [reference["roi_xyxy"]]
    )
    matched = {pair["predicted_index"] for pair in assignment["pairs"]}
    return [item for index, item in enumerate(retained) if index in matched]


def score_predictions(
    prediction_path: Path, prepared_path: Path, panel_path: Path, annotation_path: Path
) -> dict[str, Any]:
    """Seal predictions, then open exact TTC keys and score fixed denominators."""
    _, prediction_sha, predictions = snapshot_json(prediction_path)
    _, prepared_sha, prepared = snapshot_json(prepared_path)
    _, panel_sha, panel = snapshot_json(panel_path)
    query_ids = {query["garl_query_token"] for query in panel["queries"]}
    if (
        panel.get("schema") != "ttc_development_five_query_panel_v1"
        or len(panel.get("queries", ())) != 5
        or len(query_ids) != 5
        or prepared.get("schema") != "detected_phase_panel_preparation_v1"
        or predictions.get("schema") != "detected_phase_panel_predictions_v1"
        or predictions.get("status") != "EXECUTED_TRAIN40_EXPOSED_DIAGNOSTIC"
        or predictions.get("targets_read") is not False
        or predictions.get("conditions_are_fresh_and_independent") is not True
        or predictions.get("prepared_file_sha256") != prepared_sha
        or prepared.get("panel_sha256") != panel_sha
        or prepared.get("status") != "READY_FOR_RUN"
        or set(predictions.get("results", {})) != set(CONDITIONS)
        or any(set(predictions["results"][condition]) != query_ids for condition in CONDITIONS)
    ):
        raise ValueError("Executed prediction contract differs from prepared five-query panel")
    prepared_by_id = {query["query_id"]: query for query in prepared["queries"]}
    if set(prepared_by_id) != query_ids:
        raise ValueError("Prepared query denominator differs from panel")
    for condition in CONDITIONS:
        for query_id, receipt in predictions["results"][condition].items():
            expected_packets = len(prepared_by_id[query_id]["r3_label_free"]["prefix"])
            result = receipt.get("result") or {}
            if (
                receipt.get("packet_ingested_count") != expected_packets
                or result.get("schema") != "detection_phase_multiobject_v1"
                or result.get("targets_read") is not False
                or result.get("packet_ingested_count") != 1
            ):
                raise ValueError("Executed query does not attest its complete prefix ingestion")
    keys = [tuple(query["target_annotation_join_key"]) for query in panel["queries"]]
    rows = pq.read_table(
        annotation_path,
        columns=[*JOIN_KEYS, "ttc", "frame_ttc"],
        filters=[("sample_token", "in", [key[1] for key in keys])],
        use_threads=False,
    ).to_pylist()
    annotations = _exact_index(rows, JOIN_KEYS, "scoring annotation")
    truth = {}
    for query, key in zip(panel["queries"], keys, strict=True):
        row = annotations.get(key)
        if row is None or not row["frame_ttc"]:
            raise ValueError("Exact target annotation is missing")
        value = float(row["frame_ttc"][-1])
        if not math.isfinite(value) or value != float(row["ttc"]):
            raise ValueError("Target TTC is nonfinite or not the unchanged final endpoint")
        truth[query["garl_query_token"]] = value

    scored = {}
    for condition in CONDITIONS:
        errors, sign_errors = [], 0
        coverage = misses = unmatched_to_panel_reference = history_insufficient = 0
        per_query = []
        condition_results = predictions["results"].get(condition, {})
        for query in panel["queries"]:
            query_id = query["garl_query_token"]
            receipt = condition_results.get(query_id, {})
            detections = (receipt.get("result") or {}).get("detections") or []
            matched = [
                item for item in detections if item.get("evaluation_reference_match") is True
            ]
            unmatched_to_panel_reference += max(0, len(detections) - len(matched))
            history_insufficient += sum(
                item.get("ttc_reason") == "insufficient_track_history" for item in detections
            )
            available = [
                item
                for item in matched
                if item.get("ttc_status") == "available" and item.get("ttc_signed_s") is not None
            ]
            if len(available) > 1:
                raise ValueError("More than one prediction matched one target")
            if not available:
                misses += 1
                per_query.append({"query_id": query_id, "status": "MISS_OR_ABSTENTION"})
                continue
            predicted, target = float(available[0]["ttc_signed_s"]), truth[query_id]
            error = abs(predicted - target)
            errors.append(error)
            coverage += 1
            sign_error = (predicted < 0) != (target < 0)
            sign_errors += int(sign_error)
            per_query.append(
                {
                    "query_id": query_id,
                    "status": "SCORED",
                    "absolute_error_s": error,
                    "sign_error": sign_error,
                    "label_age_at_prediction_us": query["timing"]["label_age_at_prediction_us"],
                    "ttc_target_shift_applied_us": 0,
                }
            )
        scored[condition] = {
            "denominator_queries": 5,
            "coverage": coverage,
            "misses_or_abstentions": misses,
            "unmatched_to_panel_reference_count": unmatched_to_panel_reference,
            "actual_scene_false_positive_count": None,
            "actual_scene_false_positive_unavailable_reason": (
                "panel contains one target track, not exhaustive eligible frame GT/ignored rules"
            ),
            "history_insufficient": history_insufficient,
            "mae_s": float(np.mean(errors)) if errors else None,
            "median_absolute_error_s": float(np.median(errors)) if errors else None,
            "p95_absolute_error_s": float(np.percentile(errors, 95)) if errors else None,
            "sign_errors": sign_errors,
            "per_query": per_query,
        }
    return {
        "schema": "detected_phase_panel_score_v1",
        "scope": "TRAIN40_EXPOSED_DIAGNOSTIC_ONLY",
        "predictions_sha256_before_target_read": prediction_sha,
        "prepared_sha256": prepared_sha,
        "panel_sha256": panel_sha,
        "annotation_sha256": sha256_file(annotation_path),
        "target_reference": "RGB exposure start",
        "prediction_availability": "RGB exposure end",
        "ttc_target_shift_applied_us": 0,
        "phase_bounds_interpretation": "native phase only; not a continuous TTC interval",
        "reference_gt_scope": "single panel target per query; not exhaustive scene GT",
        "conditions": scored,
    }


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Use a new output path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_adapter(
    specification: str | None,
    config_path: Path | None,
) -> tuple[Callable[..., Any], Callable[..., Any], dict[str, Any]]:
    if specification is None:
        if config_path is None:
            raise ValueError("Built-in runtime requires --adapter-config")
        from operational.detection_phase_service.ttc_panel_adapter import create_panel_adapter

        adapter = create_panel_adapter(config_path)
        return adapter.runtime_factory, adapter.frame_loader, adapter.binding
    if config_path is not None:
        raise ValueError("--adapter-config is only valid with the built-in runtime adapter")
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("Runtime adapter must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    result = factory()
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Runtime adapter must return runtime_factory and frame_loader")
    module_path = Path(module.__file__).resolve() if module.__file__ else None
    if module_path is None or not module_path.is_file():
        raise ValueError("Runtime adapter must have a hashable source file")
    return (
        result[0],
        result[1],
        {
            "specification": specification,
            "source_path": str(module_path),
            "source_sha256": sha256_file(module_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--panel", type=Path, required=True)
    prepare_parser.add_argument("--predictions", type=Path, required=True)
    prepare_parser.add_argument("--eap-root", type=Path, required=True)
    prepare_parser.add_argument("--garl-root", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--prepared", type=Path, required=True)
    run_parser.add_argument("--predictions", type=Path, required=True)
    run_parser.add_argument(
        "--runtime-adapter",
        help=(
            "Optional module:function test/custom adapter; defaults to the built-in sealed adapter"
        ),
    )
    run_parser.add_argument(
        "--adapter-config",
        type=Path,
        help="Strict JSON configuration required by the built-in sealed adapter",
    )
    run_parser.add_argument("--output", type=Path, required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--prepared", type=Path, required=True)
    score_parser.add_argument("--panel", type=Path, required=True)
    score_parser.add_argument("--annotations", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_phase(args.panel, args.predictions, args.eap_root, args.garl_root)
    elif args.command == "run":
        _, prepared_file_sha, prepared = snapshot_json(args.prepared)
        _, prediction_file_sha, prediction_value = snapshot_json(args.predictions)
        detector, predictions = _validate_prediction_payload(prediction_value)
        if (
            prediction_file_sha != prepared.get("predictions_sha256")
            or detector["checkpoint_sha256"] != prepared.get("detector_checkpoint_sha256")
            or sha256_file(Path(__file__)) != prepared.get("harness_sha256")
            or sha256_file(ROOT / "operational/detection_phase_service/service.py")
            != prepared.get("phase_service_sha256")
            or sha256_file(ROOT / "scripts/replay_detection_phase_service.py")
            != prepared.get("replay_geometry_source_sha256")
            or sha256_file(ROOT / "src/e_jepa_ttc/detection/causal_tracker.py")
            != prepared.get("causal_tracker_sha256")
            or sha256_file(ROOT / "src/e_jepa_ttc/detection/geometry_contract.py")
            != prepared.get("geometry_contract_sha256")
            or sha256_file(ROOT / "operational/train40_runtime/multiobject.py")
            != prepared.get("multiobject_runtime_sha256")
        ):
            raise ValueError("Run inputs differ from the sealed preparation")
        runtime_factory, frame_loader, adapter_binding = _load_adapter(
            args.runtime_adapter, args.adapter_config
        )
        if adapter_binding.get("media_manifest_sha256") not in {
            None,
            prepared.get("media_manifest_sha256"),
        }:
            raise ValueError("Adapter eAP manifest differs from sealed preparation")
        result = run_fresh_service(prepared, predictions, runtime_factory, frame_loader)
        result.update(
            prepared_file_sha256=prepared_file_sha,
            detector_predictions_sha256=prediction_file_sha,
            detector_checkpoint_sha256=detector["checkpoint_sha256"],
            runtime_adapter=adapter_binding,
        )
    else:
        result = score_predictions(args.predictions, args.prepared, args.panel, args.annotations)
    _write_new(args.output, result)
    print(json.dumps({"status": result.get("status", "SCORED"), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
