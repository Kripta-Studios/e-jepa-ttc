"""RAW reconstruction preserves historical voxel semantics and causal boundaries."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from e_jepa_ttc.simplex_t.query_context_voxel import encode_query_window
from scripts.release_native_raw_latency import prepare


def fixture_input():
    raw = {
        "x": np.array([0, 4, 5, 9, 11, 0, 5, 6, 9], np.int32),
        "y": np.array([1, 2, 3, 4, 5, 1, 3, 6, 4], np.int32),
        "t": np.array([0, 5, 10, 10, 15, 20, 25, 29, 30], np.int64),
        "p": np.array([0, 1, 0, 1, 1, 0, 1, 0, 1], np.int8),
    }

    def chunks(start, end, **kwargs):
        mask = (raw["t"] >= start) & (raw["t"] < end)
        yield {key: value[mask] for key, value in raw.items()}

    pool = cast(
        ReaderPool, SimpleNamespace(get=lambda path: SimpleNamespace(iter_window_chunks=chunks))
    )
    item = {
        "windows_us": [[0, 10], [10, 20], [20, 30]],
        "sensor_anchor_us": 30,
        "square_xyxy": [5.0, 0.0, 15.0, 10.0],
        "raw_path": "fixture",
        "sequence_id": "s",
    }
    return raw, pool, item


def test_raw_matches_historical_per_window_encoder_with_offset_and_half_open_edges():
    raw, pool, item = fixture_input()
    actual = prepare(pool, item)
    expected = []
    for start, end in item["windows_us"]:
        mask = (raw["t"] >= start) & (raw["t"] < end)
        expected.append(
            encode_query_window(
                {k: v[mask] for k, v in raw.items()},
                square_xyxy=tuple(item["square_xyxy"]),
                start_us=start,
                end_us=end,
                sequence_id="s",
                roi_size=128,
                bins_per_polarity=5,
                event_pixel_diff=5.0,
            )
        )
    assert torch.equal(actual, torch.stack(expected)[None])
    assert actual.shape == (1, 3, 12, 128, 128)


@pytest.mark.parametrize(
    "windows,anchor",
    [
        ([[0, 10], [10, 20]], 20),
        ([[0, 10], [10, 10], [20, 30]], 30),
        ([[0, 10], [20, 30], [10, 20]], 30),
        ([[0, 10], [10, 20], [20, 30]], 29),
    ],
)
def test_invalid_or_future_sensor_windows_are_rejected(windows, anchor):
    _, pool, item = fixture_input()
    item.update(windows_us=windows, sensor_anchor_us=anchor)
    with pytest.raises(ValueError):
        prepare(pool, item)


def test_empty_crop_is_encoded_without_fabricating_events():
    _, pool, item = fixture_input()
    item["square_xyxy"] = [100.0, 100.0, 110.0, 110.0]
    value = prepare(pool, item)
    assert torch.isfinite(value).all()
    assert torch.count_nonzero(value[:, :, :10]) == 0
