"""Ingest-once chunking around the actual TRAIN40 service interfaces."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch

from e_jepa_ttc.detection.causal_tracker import CausalMultiObjectTracker, DetectionObservation
from operational.detection_phase_service import ChunkedTrain40Runtime, DetectionPhaseService
from operational.train40_runtime.multiobject import (
    MultiObjectTrain40Service,
    ObjectRegion,
    RawRGBHistory,
)


class History:
    def __init__(self) -> None:
        self.ingests: list[tuple[int, int]] = []
        self.queries: list[list[tuple[float, float, float, float]]] = []
        self.initializations: list[tuple[int, int]] = []

    def initialize(self, raw: Any, start: int, end: int) -> None:
        self.initializations.append((start, end))
        self.ingests = [(start, end)]

    def ingest(self, raw: Any, start: int, end: int) -> None:
        if self.ingests and self.ingests[-1][1] != start:
            raise ValueError("history packet gap")
        self.ingests.append((start, end))

    def query_many(self, rois):
        self.queries.append(list(rois))
        return torch.zeros((len(rois), 4, 3, 12, 2, 2), dtype=torch.float32)


class Backend:
    def __init__(self, capacity: int = 10) -> None:
        self.max_objects = capacity
        self.history = History()
        self.rgb_feature_cache: object | None = None
        self.parallel_branches = False
        self.overlap_event_with_rgb_preparation = False
        self._event_overlap_inflight = False
        self._rgb_uint8 = None
        self._closed = False
        self.packed: list[list[str]] = []
        self.inferred: list[list[str]] = []
        self.fail_chunk: int | None = None
        self.closed = 0

    def _pack_rgb(self, objects, history):
        self.packed.append([item.object_id for item in objects])
        return [[] for _ in objects]

    def infer_prepared(self, objects, query_us, event, rgb, unavailable_reason=""):
        index = len(self.inferred)
        self.inferred.append([item.object_id for item in objects])
        if self.fail_chunk == index:
            raise RuntimeError("chunk failed")
        return [
            {
                "object_id": item.object_id,
                "predictions": {
                    "PHASE_ONLY_TRAIN40_S7": -2.5,
                    "E_CTX_TRAIN40": -3.0,
                    "R_CTX_TRAIN40": -2.0,
                },
                "phase": {"point_phase": -0.04, "q10": -0.07, "q90": 0.01},
            }
            for item in objects
        ]

    def close(self):
        self.closed += 1
        self._closed = True


def regions(count: int) -> list[ObjectRegion]:
    return [
        ObjectRegion(f"track_{index}:g{index}", (index * 2.0, 0.0, index * 2.0 + 1, 1), None)
        for index in range(count)
    ]


def detections(count: int) -> list[DetectionObservation]:
    return [
        DetectionObservation(
            f"detection_{index}",
            "car",
            1.0 - index / (count + 1),
            (index * 2.0, 0.0, index * 2.0 + 1, 1),
            None,
        )
        for index in range(count)
    ]


def rgb(query_us: int) -> RawRGBHistory:
    return RawRGBHistory(query_us, {}, {}, (), "fixture_no_rgb")


@pytest.mark.parametrize(
    ("count", "chunk", "expected_chunks"),
    [(0, 4, 0), (1, 4, 1), (4, 4, 1), (7, 3, 3), (25, 10, 3)],
)
def test_chunked_runtime_ingests_once_and_returns_every_identity(
    count: int, chunk: int, expected_chunks: int
) -> None:
    backend = Backend(capacity=10)
    runtime = ChunkedTrain40Runtime(backend, max_objects=30, chunk_size=chunk)
    output = runtime.update({}, 0, 20, regions(count), rgb(20))
    assert backend.history.ingests == [(0, 20)]
    assert [len(item) for item in backend.history.queries] == [
        min(chunk, count - first) for first in range(0, count, chunk)
    ]
    assert output["packet_ingested_count"] == 1
    assert output["chunk_count"] == expected_chunks
    assert [row["object_id"] for row in output["objects"]] == [
        item.object_id for item in regions(count)
    ]


def test_actual_train40_surface_used_by_chunk_adapter_is_present() -> None:
    assert list(inspect.signature(MultiObjectTrain40Service.infer_prepared).parameters) == [
        "self",
        "objects",
        "query_us",
        "event",
        "rgb",
        "unavailable_reason",
        "_event_ticket",
    ]
    assert hasattr(MultiObjectTrain40Service, "_pack_rgb")


def test_public_history_rebase_supports_successive_irregular_query_endpoints() -> None:
    backend = Backend(capacity=4)
    runtime = ChunkedTrain40Runtime(backend, max_objects=4, chunk_size=4)
    runtime.rebase_history({}, 80_000, 680_000)
    first = runtime.update({}, 680_000, 700_000, regions(1), rgb(700_000))
    runtime.rebase_history({}, 180_000, 780_000)
    second = runtime.update({}, 780_000, 800_000, regions(1), rgb(800_000))
    assert backend.history.initializations == [(80_000, 680_000), (180_000, 780_000)]
    assert first["history_rebase_count"] == 1
    assert second["history_rebase_count"] == 2


@pytest.mark.parametrize(("start", "end"), [(False, 600_000), (0, True)])
def test_history_rebase_rejects_boolean_timestamps(start: object, end: object) -> None:
    runtime = ChunkedTrain40Runtime(Backend(capacity=4), max_objects=4, chunk_size=4)
    with pytest.raises(ValueError, match="exactly 600000"):
        runtime.rebase_history({}, start, end)  # type: ignore[arg-type]


def test_failed_history_initialization_permanently_fails_closed() -> None:
    backend = Backend(capacity=4)

    def fail_after_possible_mutation(raw: Any, start: int, end: int) -> None:
        backend.history.initializations.append((start, end))
        raise RuntimeError("initialization failed")

    backend.history.initialize = fail_after_possible_mutation  # type: ignore[method-assign]
    runtime = ChunkedTrain40Runtime(backend, max_objects=4, chunk_size=4)
    with pytest.raises(RuntimeError, match="initialization failed"):
        runtime.rebase_history({}, 0, 600_000)
    with pytest.raises(RuntimeError, match="failed"):
        runtime.rebase_history({}, 0, 600_000)
    with pytest.raises(RuntimeError, match="failed after packet ingestion"):
        runtime.update({}, 600_000, 620_000, regions(1), rgb(620_000))


def test_detection_service_processes_more_than_twenty_without_capacity_loss() -> None:
    backend = Backend(capacity=8)
    runtime = ChunkedTrain40Runtime(backend, max_objects=30, chunk_size=8)
    service = DetectionPhaseService(
        runtime,
        tracker=CausalMultiObjectTracker(confirm_hits=1),
        minimum_track_history_us=0,
    )
    report = service.update({}, 0, 20, detections(25), rgb(20))
    assert len(report["detections"]) == 25
    assert all(row["ttc_status"] == "available" for row in report["detections"])
    assert all(
        (row["phase_q10"], row["phase_q90"]) == (-0.07, 0.01) for row in report["detections"]
    )
    assert report["runtime"]["chunk_count"] == 4
    assert backend.history.ingests == [(0, 20)]
    flattened = [identity for chunk in backend.inferred for identity in chunk]
    assert len(flattened) == len(set(flattened)) == 25


def test_empty_frame_between_queries_keeps_single_contiguous_ingest() -> None:
    backend = Backend(capacity=4)
    runtime = ChunkedTrain40Runtime(backend, max_objects=30, chunk_size=4)
    service = DetectionPhaseService(
        runtime,
        tracker=CausalMultiObjectTracker(confirm_hits=1),
        minimum_track_history_us=0,
    )
    service.update({}, 0, 20, detections(1), rgb(20))
    empty = service.update({}, 20, 40, [], rgb(40))
    service.update({}, 40, 60, detections(1), rgb(60))
    assert backend.history.ingests == [(0, 20), (20, 40), (40, 60)]
    assert empty["runtime"]["chunk_count"] == 0
    assert empty["packet_ingested_count"] == 1


def test_generation_change_never_reuses_an_old_object_identity() -> None:
    backend = Backend(capacity=4)
    runtime = ChunkedTrain40Runtime(backend, max_objects=30, chunk_size=4)
    service = DetectionPhaseService(
        runtime,
        tracker=CausalMultiObjectTracker(confirm_hits=1, max_gap_us=25),
        minimum_track_history_us=0,
    )
    first = service.update({}, 0, 20, detections(1), rgb(20))["detections"][0]
    second = service.update({}, 20, 50, detections(1), rgb(50))["detections"][0]
    assert first["runtime_output"]["object_id"] != second["runtime_output"]["object_id"]
    assert backend.inferred[0][0] != backend.inferred[1][0]


def test_failure_after_ingest_fails_closed_and_cannot_duplicate_packet() -> None:
    backend = Backend(capacity=4)
    backend.fail_chunk = 1
    runtime = ChunkedTrain40Runtime(backend, max_objects=30, chunk_size=4)
    with pytest.raises(RuntimeError, match="chunk failed"):
        runtime.update({}, 0, 20, regions(7), rgb(20))
    with pytest.raises(RuntimeError, match="failed after packet ingestion"):
        runtime.update({}, 0, 20, regions(7), rgb(20))
    assert backend.history.ingests == [(0, 20)]


def test_duplicate_identity_rejected_before_packet_ingestion() -> None:
    backend = Backend(capacity=4)
    runtime = ChunkedTrain40Runtime(backend, max_objects=30, chunk_size=4)
    duplicate = regions(2)
    duplicate[1] = ObjectRegion(duplicate[0].object_id, duplicate[1].event_square, None)
    with pytest.raises(ValueError, match="unique"):
        runtime.update({}, 0, 20, duplicate, rgb(20))
    assert not backend.history.ingests


def test_historical_feature_cache_is_rejected_instead_of_leaking_across_chunks() -> None:
    backend = Backend(capacity=4)
    backend.rgb_feature_cache = object()
    with pytest.raises(ValueError, match="does not share"):
        ChunkedTrain40Runtime(backend, max_objects=30, chunk_size=4)
