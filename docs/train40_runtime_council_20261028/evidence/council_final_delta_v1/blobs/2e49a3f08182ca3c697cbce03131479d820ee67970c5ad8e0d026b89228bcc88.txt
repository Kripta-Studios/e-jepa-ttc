"""CPU fakes establish explicit synthetic priming coverage and honest failures."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from operational.train40_runtime.priming import NATIVE_RGB_LAYOUTS, prime_supported_histories


class FakeService:
    def __init__(self, max_objects=2, fail_layout=None):
        self.max_objects = max_objects
        self.device = torch.device("cpu")
        self.rgb_object_chunk_size = 1
        self.history = SimpleNamespace(
            ready=False, stats={"end_us": None, "ingested_packets": 0, "queries": 0}
        )
        self._closed = self._event_overlap_inflight = False
        self.graphs = []
        self.producers = SimpleNamespace(
            reuse_by_modality={"E": False, "R": True}, diagnostics=lambda: list(self.graphs)
        )
        self.calls = []
        self.fail_layout = fail_layout

    def infer_prepared(self, objects, query, event, rgb):
        history = len(rgb[0])
        steps = rgb[0][0][0].shape[0] if history else 0
        layout = len(objects), history, steps
        self.calls.append(layout)
        assert tuple(event.shape) == (len(objects), 4, 3, 12, 128, 128)
        assert event.dtype == torch.float32 and not bool(torch.count_nonzero(event))
        assert len(rgb) == len(objects)
        assert len({obj.object_id for obj in objects}) == len(objects)
        for obj, observations in zip(objects, rgb, strict=True):
            assert bool(observations) == (obj.rgb_square is not None)
            for pixels, metadata in observations:
                assert pixels.shape == (steps, 3, 128, 128)
                assert pixels.dtype == np.float32 and not np.any(pixels)
                stamps = metadata["timestamps_us"]
                assert stamps[-1] <= query and query - stamps[0] <= 650_000
                assert np.array_equal(np.diff(stamps), [100_000] * (steps - 1))
            for previous, current in zip(observations, observations[1:], strict=False):
                np.testing.assert_array_equal(previous[0][1:], current[0][:-1])
                assert previous[1]["timestamps_us"][1:] == current[1]["timestamps_us"][:-1]
        if layout == self.fail_layout:
            raise RuntimeError("synthetic capture failed")
        names = [("E", [len(objects) * 4, 3, 12, 128, 128], False)]
        if history:
            names.append(("R", [history, steps, 3, 128, 128], history > 1))
        for modality, shape, reuse in names:
            row = {
                "producer": modality + ("_REUSE" if reuse else ""),
                "shape": shape,
                "endpoint_reuse": reuse,
            }
            if row not in self.graphs:
                self.graphs.append(row)


@pytest.mark.parametrize("count", [1, 2])
def test_synthetic_priming_covers_native_history_abi_without_sensor_data(count):
    service = FakeService(count)
    receipt = {}
    assert prime_supported_histories(service, receipt=receipt) is receipt
    assert service.calls == [(count, history, steps) for history, steps in NATIVE_RGB_LAYOUTS]
    assert receipt["status"] == "COMPLETE"
    assert len(receipt["records"]) == 7
    assert all(row["seconds"] >= 0 for row in receipt["records"])
    assert receipt["unexpected_shapes_remain_inside_update_clock"]
    assert not receipt["numerical_parity_admitted"]
    assert not receipt["nonoverlapping_multi_observation_rgb_primed"]
    assert not receipt["event_early_launch_path_exercised"]
    rgb = [row for row in receipt["producer_graphs_after"] if row["producer"].startswith("R")]
    assert len(rgb) == 6
    assert {row["shape"][0] for row in rgb if row["endpoint_reuse"]} == {2, 3, 4, 5}
    assert {tuple(row["shape"][:2]) for row in rgb if not row["endpoint_reuse"]} == {(1, 2), (1, 3)}
    assert service.history.stats == {"end_us": None, "ingested_packets": 0, "queries": 0}


def test_synthetic_priming_can_cover_explicit_object_count_transition():
    service = FakeService(2)
    result = prime_supported_histories(service, object_counts=[1, 2])
    assert result["object_counts"] == [1, 2]
    assert len(service.calls) == 14


@pytest.mark.parametrize("state", [{"end_us": 0}, {"ingested_packets": 1}, {"queries": 1}])
def test_priming_rejects_sensor_history_before_executing_models(state):
    service = FakeService()
    service.history.stats.update(state)
    with pytest.raises(ValueError, match="must precede"):
        prime_supported_histories(service)
    assert not service.calls


@pytest.mark.parametrize("counts", [[], [True], [0], [3], [1, 1]])
def test_priming_rejects_undeclared_counts(counts):
    service = FakeService()
    with pytest.raises(ValueError, match="unique integers"):
        prime_supported_histories(service, object_counts=counts)
    assert not service.calls


def test_priming_preserves_partial_evidence_when_a_capture_fails():
    service = FakeService(1, fail_layout=(1, 2, 3))
    receipt = {}
    with pytest.raises(RuntimeError, match="synthetic capture failed"):
        prime_supported_histories(service, receipt=receipt)
    assert receipt["status"] == "FAILED"
    assert len(receipt["records"]) == 4
    assert all(row["status"] == "COMPLETE" for row in receipt["records"][:-1])
    assert receipt["records"][-1]["status"] == "FAILED"
    assert receipt["total_seconds"] > 0
    assert receipt["producer_graphs_after"]


def test_priming_does_not_overwrite_existing_receipt():
    receipt = {"status": "PREVIOUS"}
    with pytest.raises(ValueError, match="empty receipt"):
        prime_supported_histories(FakeService(), receipt=receipt)
    assert receipt == {"status": "PREVIOUS"}
