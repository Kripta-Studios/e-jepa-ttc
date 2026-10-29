"""Focused CPU tests for final TRAIN40 event-history reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from operational.final_refit.event_history import (
    _historical_pair_input,
    causal_history_mask,
    compact_referenced_observations,
    merge_fresh_pair_phase,
)


def test_historical_pair_input_preserves_v12_log_epsilon_and_support() -> None:
    token = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    support = torch.tensor([[0.2, 0.6, 0.4], [0.9, 0.1, 0.3]], dtype=torch.float32)
    elapsed = torch.tensor([0.1, 0.2], dtype=torch.float32)
    result = _historical_pair_input({"token128": token, "support": support}, elapsed)
    assert result.shape == (2, 133)
    assert torch.equal(result[:, :128], token)
    assert torch.equal(result[:, 128], elapsed)
    assert torch.equal(result[:, 129], torch.log(elapsed + 1e-8))
    assert torch.equal(result[:, 130], elapsed.reciprocal())
    assert torch.equal(result[:, 131], support[:, -1])
    assert torch.equal(result[:, 132], torch.tensor([0.4, 0.1]))


def test_merge_reuses_prefix_and_rebuilds_every_pair_dependent_coordinate() -> None:
    prefix = np.arange(20, dtype=np.float32).reshape(2, 10)
    pair = np.asarray([0.25, -0.5], dtype=np.float32)
    result = merge_fresh_pair_phase(prefix, pair)
    assert result.dtype == np.float32
    assert result.shape == (2, 17)
    np.testing.assert_array_equal(result[:, :10], prefix)
    np.testing.assert_array_equal(result[:, 10], pair)
    expected = np.stack(
        (pair - prefix[:, 8], pair - prefix[:, 9], prefix[:, 9] - prefix[:, 8]), axis=-1
    )
    np.testing.assert_array_equal(result[:, 11:14], expected)
    np.testing.assert_array_equal(result[:, 14:17], np.abs(expected))


@pytest.mark.parametrize(
    ("prefix", "pair"),
    [
        (np.zeros((2, 9), np.float32), np.zeros(2, np.float32)),
        (np.zeros((2, 10), np.float32), np.zeros(1, np.float32)),
        (np.full((2, 10), np.nan, np.float32), np.zeros(2, np.float32)),
    ],
)
def test_merge_rejects_shape_or_finiteness_drift(prefix: np.ndarray, pair: np.ndarray) -> None:
    with pytest.raises(ValueError):
        merge_fresh_pair_phase(prefix, pair)


def test_causal_history_mask_trims_only_the_oldest_complete_inputs() -> None:
    anchors = [100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000]
    result = causal_history_mask(anchors, query_us=470_000, input_span_us=300_000)
    np.testing.assert_array_equal(
        result, np.asarray([False, True, True, True, True, True, True, True])
    )


def test_causal_history_mask_rejects_nonchronological_clocks() -> None:
    with pytest.raises(ValueError, match="clocks"):
        causal_history_mask([100, 100], query_us=200, input_span_us=50)


def test_compaction_prunes_unused_observation_and_preserves_aligned_rows() -> None:
    source = {
        "observation_id": np.asarray(["o0", "unused", "o2"]),
        "features": np.asarray([[1, 2], [3, 4], [5, 6]], np.float32),
        "anchor_us": np.asarray([10, 20, 30], np.int64),
        "group_id": np.asarray(["g0", "g0", "g1"]),
    }
    result = compact_referenced_observations(
        source,
        np.asarray([["o0", ""], ["", "o2"]]),
        np.asarray([[True, False], [False, True]]),
    )
    np.testing.assert_array_equal(result["observation_id"], ["o0", "o2"])
    np.testing.assert_array_equal(result["features"], [[1, 2], [5, 6]])
    np.testing.assert_array_equal(result["anchor_us"], [10, 30])
    np.testing.assert_array_equal(result["group_id"], ["g0", "g1"])


def test_compaction_rejects_referenced_missing_observation() -> None:
    source = {
        "observation_id": np.asarray(["o0"]),
        "features": np.asarray([[1, 2]], np.float32),
    }
    with pytest.raises(ValueError, match="missing generated observations"):
        compact_referenced_observations(source, np.asarray([["missing"]]), np.asarray([[True]]))
