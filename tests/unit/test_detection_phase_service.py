from __future__ import annotations

import inspect
from typing import Literal

import pytest

from e_jepa_ttc.detection.causal_tracker import CausalMultiObjectTracker, DetectionObservation
from operational.detection_phase_service import DetectionPhaseService
from operational.train40_runtime.multiobject import MultiObjectTrain40Service, RawRGBHistory


class History:
    def __init__(self) -> None:
        self.calls = []

    def ingest(self, raw, start, end):
        self.calls.append((raw, start, end))


class Runtime:
    def __init__(self, max_objects: int = 20) -> None:
        self.max_objects = max_objects
        self.history = History()
        self.calls = []
        self.closed = 0
        self.rgb_object_chunk_size: int | None = None

    def update(self, raw_packet, start_us, end_us, objects, rgb_history):
        self.calls.append((raw_packet, start_us, end_us, objects, rgb_history))
        self.history.ingest(raw_packet, start_us, end_us)
        return {
            "objects": [
                {
                    "object_id": item.object_id,
                    "predictions": {"PHASE_ONLY_TRAIN40_S7": -2.5},
                    "phase": {"point_phase": -0.04, "q10": -0.07, "q90": -0.02},
                }
                for item in objects
            ],
            "object_count": len(objects),
        }

    def close(self):
        self.closed += 1


def rgb(query: int) -> RawRGBHistory:
    return RawRGBHistory(query, {}, {}, (), "fixture")


def detected(
    identity: str,
    left: float = 0.0,
    source: Literal["measured", "extrapolated"] = "measured",
):
    box = (left, 0.0, left + 10.0, 10.0)
    return DetectionObservation(identity, "car", 0.9, box, box, source)


def test_adapter_matches_actual_train40_update_signature() -> None:
    parameters = list(inspect.signature(MultiObjectTrain40Service.update).parameters)
    assert parameters == ["self", "raw_packet", "start_us", "end_us", "objects", "rgb_history"]


def test_tentative_frame_ingests_once_then_confirmed_frame_runs_all_objects() -> None:
    runtime = Runtime()
    service = DetectionPhaseService(runtime, minimum_track_history_us=0)
    first = service.update({}, 0, 20, [detected("d1")], rgb(20))
    assert first["packet_ingested_count"] == 1
    assert len(runtime.history.calls) == 1 and not runtime.calls
    assert first["detections"][0]["ttc_reason"] == "track_not_confirmed"

    second = service.update({}, 20, 40, [detected("d2", 0.5)], rgb(40))
    assert len(runtime.calls) == 1
    assert len(runtime.history.calls) == 2
    result = second["detections"][0]
    assert result["ttc_status"] == "available"
    assert result["ttc_signed_s"] == -2.5
    assert (result["phase_q10"], result["phase_q90"]) == (-0.07, -0.02)
    assert result["phase_bounds_semantics"].startswith("native_phase_quantile")


def test_capacity_and_extrapolation_abstain_but_retain_all_detections() -> None:
    runtime = Runtime(max_objects=1)
    tracker = CausalMultiObjectTracker(confirm_hits=1)
    service = DetectionPhaseService(runtime, tracker=tracker, minimum_track_history_us=0)
    report = service.update(
        {},
        0,
        20,
        [
            detected("best", 0),
            DetectionObservation("lower", "car", 0.4, (20, 0, 30, 10), (20, 0, 30, 10)),
            detected("predicted", 40, "extrapolated"),
        ],
        rgb(20),
    )
    assert len(report["detections"]) == 3
    by_id = {item["detection_id"]: item for item in report["detections"]}
    assert by_id["best"]["ttc_status"] == "available"
    assert by_id["lower"]["ttc_reason"] == "runtime_object_capacity"
    assert by_id["predicted"]["ttc_reason"] == "roi_extrapolated"
    assert len(runtime.calls) == 1 and len(runtime.calls[0][3]) == 1


def test_runtime_chunking_receives_several_objects_in_one_ingest() -> None:
    runtime = Runtime(max_objects=4)
    runtime.rgb_object_chunk_size = 1
    service = DetectionPhaseService(
        runtime,
        tracker=CausalMultiObjectTracker(confirm_hits=1),
        minimum_track_history_us=0,
    )
    report = service.update(
        {}, 0, 20, [detected("a", 0), detected("b", 20), detected("c", 40)], rgb(20)
    )
    assert len(runtime.calls) == 1 and len(runtime.history.calls) == 1
    assert len(runtime.calls[0][3]) == 3
    assert all(item["ttc_status"] == "available" for item in report["detections"])


def test_full_detector_geometry_and_velocity_semantics_are_preserved() -> None:
    runtime = Runtime()
    service = DetectionPhaseService(
        runtime,
        tracker=CausalMultiObjectTracker(confirm_hits=1),
        minimum_track_history_us=0,
    )
    observation = DetectionObservation(
        "d",
        "truck",
        0.7,
        (0, 0, 10, 10),
        (1, 2, 9, 9),
        center_3d_m=(1.0, 2.0, 30.0),
        size_lwh_m=(4.0, 2.0, 2.5),
        yaw_rad=-0.4,
        velocity=(3.0, -0.2),
        velocity_frame="released_native_components",
        velocity_semantics="published_first_two_components_not_ttc",
    )
    result = service.update({}, 0, 20, [observation], rgb(20))["detections"][0]
    assert result["center_3d_m"] == (1.0, 2.0, 30.0)
    assert result["size_lwh_m"] == (4.0, 2.0, 2.5)
    assert result["yaw_rad"] == -0.4
    assert result["velocity"] == (3.0, -0.2)
    assert result["velocity_frame"] == "released_native_components"


def test_missing_declared_final_ttc_abstains_without_using_branch_value() -> None:
    runtime = Runtime()

    def branches_only(raw_packet, start_us, end_us, objects, rgb_history):
        runtime.history.ingest(raw_packet, start_us, end_us)
        return {
            "objects": [
                {
                    "object_id": item.object_id,
                    "predictions": {"E_CTX_TRAIN40": 1.0, "R_CTX_TRAIN40": 2.0},
                }
                for item in objects
            ]
        }

    runtime.update = branches_only
    service = DetectionPhaseService(
        runtime,
        tracker=CausalMultiObjectTracker(confirm_hits=1),
        minimum_track_history_us=0,
    )
    result = service.update({}, 0, 20, [detected("a")], rgb(20))["detections"][0]
    assert result["ttc_status"] == "unavailable"
    assert result["ttc_reason"] == "runtime_missing_declared_final_ttc"
    assert result["ttc_signed_s"] is None


def test_default_track_history_gate_prevents_early_phase_query() -> None:
    runtime = Runtime()
    tracker = CausalMultiObjectTracker(confirm_hits=1)
    service = DetectionPhaseService(runtime, tracker=tracker)
    result = service.update({}, 0, 20_000, [detected("a")], rgb(20_000))["detections"][0]
    assert result["ttc_reason"] == "insufficient_track_history"
    assert not runtime.calls and len(runtime.history.calls) == 1


def test_gap_creates_new_generation_and_cannot_inherit_phase_identity() -> None:
    runtime = Runtime()
    tracker = CausalMultiObjectTracker(confirm_hits=1, max_gap_us=25)
    service = DetectionPhaseService(runtime, tracker=tracker, minimum_track_history_us=0)
    first = service.update({}, 0, 20, [detected("a")], rgb(20))["detections"][0]
    second = service.update({}, 20, 50, [detected("b")], rgb(50))["detections"][0]
    assert (first["track_id"], first["generation"]) != (
        second["track_id"],
        second["generation"],
    )
    assert first["runtime_output"]["object_id"] != second["runtime_output"]["object_id"]


def test_ambiguous_identity_is_retained_and_never_sent_to_phase() -> None:
    runtime = Runtime()
    tracker = CausalMultiObjectTracker(confirm_hits=1, ambiguity_margin=0.05)
    service = DetectionPhaseService(runtime, tracker=tracker, minimum_track_history_us=0)
    service.update({}, 0, 20, [detected("old")], rgb(20))
    report = service.update(
        {},
        20,
        40,
        [
            DetectionObservation("left", "car", 0.9, (0, 0, 6, 10), (0, 0, 6, 10)),
            DetectionObservation("right", "car", 0.8, (4, 0, 10, 10), (4, 0, 10, 10)),
        ],
        rgb(40),
    )
    ambiguous = [item for item in report["detections"] if item["association_status"] == "ambiguous"]
    assert len(ambiguous) == 2
    assert all(item["ttc_status"] == "unavailable" for item in ambiguous)
    assert all(item["ttc_reason"].startswith("identity_ambiguous") for item in ambiguous)
    admitted_ids = {item.object_id for item in runtime.calls[-1][3]}
    ambiguous_object_ids = {f"{item['track_id']}:g{item['generation']}" for item in ambiguous}
    assert ambiguous_object_ids.isdisjoint(admitted_ids)


def test_packet_replay_and_close_are_rejected_or_idempotent() -> None:
    runtime = Runtime()
    service = DetectionPhaseService(
        runtime,
        tracker=CausalMultiObjectTracker(confirm_hits=1),
        minimum_track_history_us=0,
    )
    service.update({}, 0, 20, [detected("a")], rgb(20))
    with pytest.raises(ValueError, match="contiguous"):
        service.update({}, 0, 20, [detected("b")], rgb(20))
    service.close()
    service.close()
    assert runtime.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        service.update({}, 20, 40, [detected("c")], rgb(40))


def test_malformed_result_after_ingest_fails_closed_against_packet_retry() -> None:
    runtime = Runtime()

    def malformed(raw_packet, start_us, end_us, objects, rgb_history):
        runtime.history.ingest(raw_packet, start_us, end_us)
        return {"objects": []}

    runtime.update = malformed
    service = DetectionPhaseService(
        runtime,
        tracker=CausalMultiObjectTracker(confirm_hits=1),
        minimum_track_history_us=0,
    )
    with pytest.raises(ValueError, match="result count"):
        service.update({}, 0, 20, [detected("a")], rgb(20))
    with pytest.raises(RuntimeError, match="failed after packet ownership"):
        service.update({}, 0, 20, [detected("a")], rgb(20))
    assert len(runtime.history.calls) == 1
    service.close()
    assert runtime.closed == 1
