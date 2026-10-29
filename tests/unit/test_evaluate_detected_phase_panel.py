"""Focused contracts for the four-condition detected-PHASE panel harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_detected_phase_panel as panel_eval


def test_roi_perturbation_preserves_extent_and_stays_inside_image() -> None:
    assert panel_eval.perturb_roi([100, 100, 300, 200]) == [110, 105, 310, 205]
    shifted = panel_eval.perturb_roi([1250, 700, 1279, 719])
    assert shifted == pytest.approx([1251, 700.95, 1280, 719.95])
    assert shifted[2] - shifted[0] == pytest.approx(29)
    assert shifted[3] - shifted[1] == pytest.approx(19)


def test_detector_artifact_requires_label_free_checkpoint_bound_rows(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint_sha256": "a" * 64,
                "inference_source_sha256": "b" * 64,
                "ground_truth_in_output": False,
                "predictions": {"token": [{"score": 0.5, "target": 1.0}]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contains supervision"):
        panel_eval._prediction_payload(path)


def test_snapshot_json_carries_original_bytes_after_path_mutation(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"value":1}', encoding="utf-8")
    original_bytes, original_hash, value = panel_eval.snapshot_json(path)
    path.write_text('{"value":2}', encoding="utf-8")
    assert original_bytes == b'{"value":1}'
    assert original_hash != panel_eval.sha256_file(path)
    assert value == {"value": 1}


def test_runtime_identity_registry_rejects_reuse_without_retaining_dead_objects() -> None:
    class Runtime:
        pass

    runtime = Runtime()
    references = []
    panel_eval.register_fresh_runtime(runtime, references)
    with pytest.raises(ValueError, match="fresh runtime"):
        panel_eval.register_fresh_runtime(runtime, references)


def prepared(status: str = "READY_FOR_RUN") -> dict:
    frames = [
        {
            "sample_token": f"t{index}",
            "exposure_start_us": index * 100_000,
            "exposure_end_us": (index + 1) * 100_000,
        }
        for index in range(7)
    ]
    reference = [
        {
            "sample_token": f"t{index}",
            "reference_id": "id",
            "roi_xyxy": [0, 0, 10, 10],
        }
        for index in range(7)
    ]
    return {
        "status": status,
        "queries": [
            {
                "query_id": "query",
                "sequence_id": "sequence",
                "r3_label_free": {"prefix": frames},
                "oracle": {"R0": reference, "R1": reference},
            }
        ],
    }


def test_run_uses_fresh_service_and_ingests_every_packet_once_including_empty(
    monkeypatch,
) -> None:
    runtimes = []

    class Runtime:
        max_objects = 8
        supports_empty_updates = True

        def __init__(self, condition) -> None:
            self.calls = []
            self.closed = False
            self.condition = condition
            self.history = object()

        def update(self, packet, start, end, objects, rgb):
            self.calls.append((start, end, len(objects)))
            rows = [
                {
                    "object_id": item.object_id,
                    "predictions": {"PHASE_ONLY_TRAIN40_S7": -1.25},
                    "phase": {"point_phase": 0.1, "q10": 0.05, "q90": 0.2},
                }
                for item in objects
            ]
            return {"objects": rows, "packet_ingested_count": 1, "targets_read": False}

        def close(self):
            self.closed = True

    def factory(condition, query):
        if condition == "R3":
            assert "oracle" not in query
        runtime = Runtime(condition)
        runtimes.append(runtime)
        return runtime

    def frame_loader(condition, query, frame):
        return {"t": np.array([], dtype=np.int64)}, object()

    def detections(condition, query, frame, rows):
        return [
            panel_eval.DetectionObservation(
                detection_id=f"{condition}:{frame['sample_token']}",
                class_name="car" if condition == "R3" else "reference_object_oracle",
                score=1.0,
                event_square=(0, 0, 10, 10),
                rgb_square=(0, 0, 10, 10),
            )
        ]

    monkeypatch.setattr(panel_eval, "build_condition_detections", detections)
    prediction_rows = {f"t{index}": [{}] for index in range(7)}
    result = panel_eval.run_fresh_service(prepared(), prediction_rows, factory, frame_loader)
    assert len(runtimes) == 4
    assert all(len(runtime.calls) == 7 for runtime in runtimes)
    assert all(runtime.closed for runtime in runtimes)
    for condition in panel_eval.CONDITIONS:
        receipt = result["results"][condition]["query"]
        assert receipt["packet_ingested_count"] == 7
        item = receipt["result"]["detections"][0]
        assert item["evaluation_reference_match"] is True
        assert item["track_id"] == ("track_000001" if condition == "R3" else "id")
        assert item["phase_q10"] == 0.05 and item["phase_q90"] == 0.2
    assert result["phase_bounds_semantics"].startswith("native q10/q90")


def test_run_refuses_incomplete_prefix_prediction_coverage() -> None:
    def unused_loader(condition, query, frame):
        return {"t": np.array([], dtype=np.int64)}, object()

    with pytest.raises(ValueError, match="Exact detector prefix coverage"):
        panel_eval.run_fresh_service(
            prepared("BLOCKED_PREDICTION_COVERAGE"),
            {},
            lambda *_: None,
            unused_loader,
        )


def test_r2_hungarian_oracle_selects_detector_roi_before_reference_tracking(
    monkeypatch,
) -> None:
    observations = [
        panel_eval.DetectionObservation("near", "car", 0.9, (0, 0, 10, 10)),
        panel_eval.DetectionObservation("far", "car", 0.8, (30, 30, 40, 40)),
    ]
    monkeypatch.setattr(
        panel_eval,
        "project_detection",
        lambda row, **kwargs: observations[row["index"]],
    )
    query = {
        "oracle": {
            "R2": {"reference_rois": [{"sample_token": "token", "roi_xyxy": [0, 0, 10, 10]}]}
        }
    }
    frame = {
        "sample_token": "token",
        "K_event": np.eye(3).tolist(),
        "T_event_ego": np.eye(4).tolist(),
    }
    result = panel_eval.build_condition_detections("R2", query, frame, [{"index": 0}, {"index": 1}])
    assert [item.detection_id for item in result] == ["near"]


def test_score_seals_predictions_before_exact_target_read(tmp_path: Path, monkeypatch) -> None:
    queries = []
    annotation_rows = []
    condition_results = {condition: {} for condition in panel_eval.CONDITIONS}
    for index in range(5):
        query_id = f"query-{index}"
        key = ["sequence", query_id, "track", "public", index]
        queries.append(
            {
                "garl_query_token": query_id,
                "target_annotation_join_key": key,
                "timing": {"label_age_at_prediction_us": 1700},
            }
        )
        annotation_rows.append(
            {
                **dict(zip(panel_eval.JOIN_KEYS, key, strict=True)),
                "ttc": -2.0,
                "frame_ttc": [-3.0, -2.0],
            }
        )
        for condition in panel_eval.CONDITIONS:
            condition_results[condition][query_id] = {
                "packet_ingested_count": 2,
                "result": {
                    "schema": "detection_phase_multiobject_v1",
                    "targets_read": False,
                    "packet_ingested_count": 1,
                    "detections": [
                        {
                            "evaluation_reference_match": True,
                            "ttc_status": "available",
                            "ttc_reason": "",
                            "ttc_signed_s": -1.5,
                            "phase_q10": -0.2,
                            "phase_q90": 0.2,
                        }
                    ],
                },
            }
    panel_path = tmp_path / "panel.json"
    prediction_path = tmp_path / "predictions.json"
    prepared_path = tmp_path / "prepared.json"
    annotation_path = tmp_path / "annotations.parquet"
    panel_path.write_text(
        json.dumps({"schema": "ttc_development_five_query_panel_v1", "queries": queries}),
        encoding="utf-8",
    )
    prepared_path.write_text(
        json.dumps(
            {
                "schema": "detected_phase_panel_preparation_v1",
                "status": "READY_FOR_RUN",
                "panel_sha256": panel_eval.sha256_file(panel_path),
                "queries": [
                    {
                        "query_id": query["garl_query_token"],
                        "r3_label_free": {"prefix": [{}, {}]},
                    }
                    for query in queries
                ],
            }
        ),
        encoding="utf-8",
    )
    prediction_path.write_text(
        json.dumps(
            {
                "schema": "detected_phase_panel_predictions_v1",
                "status": "EXECUTED_TRAIN40_EXPOSED_DIAGNOSTIC",
                "targets_read": False,
                "conditions_are_fresh_and_independent": True,
                "prepared_file_sha256": panel_eval.sha256_file(prepared_path),
                "results": condition_results,
            }
        ),
        encoding="utf-8",
    )
    annotation_path.write_bytes(b"bound-annotation-bytes")
    sealed = panel_eval.sha256_file(prediction_path)

    class Table:
        def to_pylist(self):
            assert panel_eval.sha256_file(prediction_path) == sealed
            return annotation_rows

    monkeypatch.setattr(panel_eval.pq, "read_table", lambda *args, **kwargs: Table())
    result = panel_eval.score_predictions(
        prediction_path, prepared_path, panel_path, annotation_path
    )
    assert result["predictions_sha256_before_target_read"] == sealed
    assert result["ttc_target_shift_applied_us"] == 0
    for condition in panel_eval.CONDITIONS:
        metric = result["conditions"][condition]
        assert metric["denominator_queries"] == 5
        assert metric["coverage"] == 5
        assert metric["mae_s"] == 0.5
        assert metric["sign_errors"] == 0
        assert metric["actual_scene_false_positive_count"] is None
        assert metric["unmatched_to_panel_reference_count"] == 0
    assert result["phase_bounds_interpretation"].endswith("not a continuous TTC interval")
