"""Union chunking must preserve the exact independently encoded event tensors."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union
from e_jepa_ttc.simplex_t.query_context_voxel import encode_query_window


def fixture():
    random = np.random.default_rng(7)
    raw = {
        "x": random.integers(-10, 200, 1000, dtype=np.int32),
        "y": random.integers(-10, 200, 1000, dtype=np.int32),
        "t": np.arange(1000, dtype=np.int64) * 1000,
        "p": random.integers(0, 2, 1000, dtype=np.int8),
    }
    reader = Mock()
    reader.iter_window_chunks.return_value = [
        {k: v[i : i + 100] for k, v in raw.items()} for i in range(0, 1000, 100)
    ]
    windows = np.array([[750000, 800000], [850000, 900000], [950000, 1000000]])
    lag = np.arange(15, -1, -1) * 50000
    return reader, raw, windows, lag


def test_union_tensor_is_bit_identical_to_each_window():
    reader, raw, windows, lag = fixture()
    square = (0.0, 0.0, 100.0, 100.0)
    actual = encode_context_union(
        reader,
        windows,
        lag,
        np.ones(16, bool),
        square,
        sequence_id="fixture",
        roi_size=16,
        event_pixel_diff=5.0,
    )
    for slot in range(16):
        for w, (start, end) in enumerate(windows - lag[slot]):
            keep = (raw["t"] >= start) & (raw["t"] < end)
            expected = encode_query_window(
                {k: v[keep] for k, v in raw.items()},
                square_xyxy=square,
                start_us=int(start),
                end_us=int(end),
                sequence_id="fixture",
                roi_size=16,
                bins_per_polarity=5,
                event_pixel_diff=5.0,
            )
            assert torch.equal(actual[slot, w], expected)


def test_union_memory_cap_fails_without_substituting_windows():
    reader, _, windows, lag = fixture()
    with pytest.raises(RuntimeError, match="RESOURCE_PAUSE"):
        encode_context_union(
            reader,
            windows,
            lag,
            np.ones(16, bool),
            (0.0, 0.0, 100.0, 100.0),
            sequence_id="fixture",
            roi_size=16,
            event_pixel_diff=5.0,
            retained_bytes_max=1,
        )


def test_bounded_window_reads_equal_union_without_dropping_events():
    reader, raw, windows, lag = fixture()

    def chunks(start, end, *, chunk_events):
        keep = (raw["t"] >= start) & (raw["t"] < end)
        selected = {key: values[keep] for key, values in raw.items()}
        return [selected]

    reader.iter_window_chunks.side_effect = chunks
    kwargs = dict(sequence_id="fixture", roi_size=16, event_pixel_diff=5.0)
    valid = np.ones(16, bool)
    valid[:2] = False
    expected = encode_context_union(reader, windows, lag, valid, (0.0, 0.0, 100.0, 100.0), **kwargs)
    reader.reset_mock()
    actual = encode_context_union(
        reader, windows, lag, valid, (0.0, 0.0, 100.0, 100.0), retained_bytes_max=1000, **kwargs
    )
    assert reader.iter_window_chunks.call_count == 1 + 3 * int(valid.sum())
    assert torch.equal(actual, expected)
    assert not actual[:2].any()
