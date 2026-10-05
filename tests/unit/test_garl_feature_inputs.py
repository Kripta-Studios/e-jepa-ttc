"""Native inference preparation is sensor-only, read-only and exact in CPU workers."""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from operational.efficient_context import garl_feature_inputs, parallel_inputs
from operational.efficient_context.garl_feature_inputs import prepare_inference


def inputs(tmp_path, monkeypatch):
    from e_jepa_ttc.efficient_context import garl_input

    path = tmp_path / "s/events.h5"
    path.parent.mkdir()
    path.write_bytes(b"fixture raw identity")
    stat = path.stat()
    native = {
        "sequence_id": "s",
        "boxes_xyxy": [[0, 0, 4, 4], [0, 0, 4, 4]],
        "event_windows_us": [[0, 100000], [100000, 200000]],
    }
    array = np.arange(40 * 3 * 3, dtype=np.float32).reshape(40, 3, 3)
    calls = []
    pool = object()
    monkeypatch.setattr(parallel_inputs, "_pool", pool)

    def encode(row, readers, root):
        assert set(row) == {"sequence_id", "boxes_xyxy", "event_windows_us"}
        assert readers is pool
        assert root == tmp_path
        calls.append(row)
        return torch.from_numpy(array.copy())

    monkeypatch.setattr(garl_input, "inference_record", encode)
    return native, (stat.st_size, stat.st_mtime_ns), array, path, calls


def test_original_encoder_and_fp32_output_used_without_extra_fields(tmp_path, monkeypatch):
    native, identity, array, _, calls = inputs(tmp_path, monkeypatch)
    actual = prepare_inference(native, str(tmp_path), identity)
    assert np.array_equal(actual, array)
    assert actual.dtype == np.float32
    assert calls == [native]


@pytest.mark.parametrize(
    "label", ["target", "frame_ttc", "box3d_Fcam", "box3d_h", "scenario", "producer"]
)
def test_worker_refuses_labels_geometry_or_producer_outputs(tmp_path, monkeypatch, label):
    native, identity, _, _, calls = inputs(tmp_path, monkeypatch)
    native[label] = 1
    with pytest.raises(ValueError, match="sensor/ROI/time fields only"):
        prepare_inference(native, str(tmp_path), identity)
    assert not calls


def test_changed_raw_before_prepare_refused(tmp_path, monkeypatch):
    native, identity, _, path, calls = inputs(tmp_path, monkeypatch)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="before native inference"):
        prepare_inference(native, str(tmp_path), identity)
    assert not calls


def test_changed_raw_during_prepare_refused(tmp_path, monkeypatch):
    from e_jepa_ttc.efficient_context import garl_input

    native, identity, array, path, _ = inputs(tmp_path, monkeypatch)

    def change(row, readers, root):
        path.write_bytes(b"changed")
        return torch.from_numpy(array)

    monkeypatch.setattr(garl_input, "inference_record", change)
    with pytest.raises(ValueError, match="during native inference"):
        prepare_inference(native, str(tmp_path), identity)


def test_uninitialized_reader_refused(tmp_path, monkeypatch):
    native, identity, _, _, calls = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(parallel_inputs, "_pool", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        prepare_inference(native, str(tmp_path), identity)
    assert not calls


def test_host_ram_floor_preserved(tmp_path, monkeypatch):
    import psutil

    native, identity, _, _, calls = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=2 * 1024**3 - 1)
    )
    with pytest.raises(InterruptedError, match="2GiB"):
        prepare_inference(native, str(tmp_path), identity)
    assert not calls


def test_submission_is_cpu_only_read_only_and_retains_source_worker_initializer(
    tmp_path, monkeypatch
):
    calls = []
    future = Future()

    class Executor:
        def __init__(self, *, max_workers, initializer):
            assert max_workers == 4
            assert initializer is parallel_inputs.worker_init

        def submit(self, function, native, root, identity):
            calls.append((function, native, root, identity))
            return future

        def shutdown(self, *, wait, cancel_futures):
            assert wait and cancel_futures
            calls.append("closed")

    monkeypatch.setattr(garl_feature_inputs, "ProcessPoolExecutor", Executor)
    c = SimpleNamespace(raw=tmp_path, require_resources=lambda: calls.append("resources"))
    workers = garl_feature_inputs.InferenceWorkers(c)
    native = {"sequence_id": "s", "boxes_xyxy": [], "event_windows_us": []}
    assert workers.submit(native, (3, 4)) is future
    assert calls == ["resources", (prepare_inference, native, str(tmp_path), (3, 4))]
    workers.close()
    assert calls[-1] == "closed"
