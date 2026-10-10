"""Real inference preservation and total-context contract checks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from operational.rgb_port.infer_experts import (
    Fragments,
    bounded_event_history,
    clock_features,
    delta_matrix,
)


def test_total_context_includes_expert_input_span() -> None:
    anchors = [index * 100000 for index in range(9)]
    timeline = [(anchor, str(index)) for index, anchor in enumerate(anchors)]
    observations = {identity: {"input_span_us": 300005} for _, identity in timeline}
    assert bounded_event_history(timeline, anchors, observations, 800000) == ["5", "6", "7", "8"]
    assert bounded_event_history(timeline, anchors, observations, 0) == ["0"]


def test_context_excludes_future_and_oversized_inputs() -> None:
    timeline = [(0, "a"), (100000, "b"), (200000, "future")]
    observations = {"a": {"input_span_us": 700000}, "b": {"input_span_us": 300000}}
    assert bounded_event_history(timeline, [0, 100000, 200000], observations, 100000) == ["b"]
    with pytest.raises(ValueError, match="650 ms"):
        bounded_event_history(timeline, [0, 100000, 200000], observations, 0)


def test_true_rgb_intervals_are_not_scalar_expanded() -> None:
    measured = torch.tensor([[0.100002, 0.099999]])
    assert torch.equal(delta_matrix(measured, 3), measured)
    assert torch.equal(delta_matrix(torch.tensor([0.1]), 3), torch.tensor([[0.1, 0.1]]))
    with pytest.raises(ValueError, match="shape"):
        delta_matrix(measured, 2)


def test_clock_features_subtract_integer_timestamps_and_reject_future() -> None:
    base = 2**55
    actual = clock_features([base, base + 100001], [base + 3, base + 100005], base + 200010)
    np.testing.assert_allclose(actual[:, 0], [0.200010, 0.100009], atol=1e-8)
    np.testing.assert_allclose(actual[:, 3], [0.000003, 0.000004], atol=1e-10)
    with pytest.raises(ValueError, match="unavailable"):
        clock_features([0], [11], 10)


def test_fragments_recover_bytes_and_reject_changed_identity(tmp_path: Path) -> None:
    store = Fragments(tmp_path, {"parent": "fixed", "role": "H"})
    assert store.read(0) is None
    values = {"phase": np.asarray([1.0, -2.0], np.float32)}
    store.write(0, values)
    recovered = Fragments(tmp_path, {"parent": "fixed", "role": "H"}).read(0)
    assert recovered is not None
    np.testing.assert_array_equal(recovered["phase"], values["phase"])
    with pytest.raises(ValueError, match="identity changed"):
        Fragments(tmp_path, {"parent": "different", "role": "H"})
    part = tmp_path / "part_000000.npz"
    part.write_bytes(part.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs"):
        store.read(0)
