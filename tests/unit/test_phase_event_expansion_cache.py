"""Causal, order-preserving cache selection independent of model targets."""

import numpy as np
import pytest

from scripts.cache_phase_event_expansion_development import (
    pad_event_roi,
    select_roi_events,
    trailing_window,
    uniform_indices,
)


def test_trailing_window_clips_to_sensor_coverage_and_includes_query_tick() -> None:
    assert trailing_window(40_081, 0, 3_086_051) == (0, 40_082)
    assert trailing_window(200_000, 3, 3_086_051) == (100_000, 200_001)
    assert trailing_window(3_086_313, 0, 3_086_051) == (2_986_313, 3_086_051)


def test_current_roi_padding_is_fixed_and_clipped_to_resolution() -> None:
    np.testing.assert_allclose(pad_event_roi((10, 20, 30, 40), (100, 80)), (8, 18, 32, 42))
    np.testing.assert_allclose(pad_event_roi((0, 0, 100, 80), (100, 80)), (0, 0, 100, 80))


def test_causal_selection_rejects_future_input_and_preserves_half_open_roi() -> None:
    events = np.array([[1, 1, 10, 0], [2, 2, 20, 1], [3, 3, 30, 0]], dtype=np.int64)
    chosen, count, _ = select_roi_events(events, 10, 31, 30, (1, 1, 3, 3), 20)
    assert count == 2
    np.testing.assert_array_equal(chosen, events[:2])
    with pytest.raises(ValueError, match="causal"):
        select_roi_events(events, 10, 31, 29, (0, 0, 4, 4), 20)
    with pytest.raises(ValueError, match="window"):
        select_roi_events(events, 11, 31, 30, (0, 0, 4, 4), 20)


def test_uniform_selection_is_deterministic_chronological_and_preserves_polarity() -> None:
    events = np.column_stack(
        (np.ones(101), np.ones(101), np.arange(101), np.arange(101) % 2)
    ).astype(np.int64)
    result, count, indices = select_roi_events(events, 0, 101, 100, (0, 0, 2, 2), 20)
    second, _, _ = select_roi_events(events, 0, 101, 100, (0, 0, 2, 2), 20)
    assert count == 101 and len(result) == 20
    assert len(set(indices)) == 20 and indices[0] == 0 and indices[-1] == 100
    np.testing.assert_array_equal(result, events[indices])
    np.testing.assert_array_equal(result, second)
    assert np.all(np.diff(result[:, 2]) >= 0)


def test_empty_and_small_populations_have_no_fabricated_events() -> None:
    empty = np.empty((0, 4), dtype=np.int64)
    selected, count, indices = select_roi_events(empty, 0, 100, 99, (0, 0, 2, 2), 20)
    assert selected.shape == (0, 4) and count == 0 and len(indices) == 0
    np.testing.assert_array_equal(uniform_indices(4, 20), np.arange(4))
    with pytest.raises(ValueError, match="positive"):
        uniform_indices(4, 0)


def test_unsorted_and_noninteger_events_are_rejected() -> None:
    events = np.array([[1, 1, 20, 0], [1, 1, 10, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="chronological"):
        select_roi_events(events, 0, 30, 29, (0, 0, 2, 2), 20)
    with pytest.raises(ValueError, match="integer"):
        select_roi_events(events.astype(float), 0, 30, 29, (0, 0, 2, 2), 20)
