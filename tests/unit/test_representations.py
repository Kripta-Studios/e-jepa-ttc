import numpy as np

from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations import (
    encode_event_count,
    encode_sparse_tokens,
    encode_time_surface,
    encode_voxel_grid,
)


def _events() -> EventBatch:
    return EventBatch(
        x=np.array([1, 1, 2, 3], dtype=np.int32),
        y=np.array([1, 1, 2, 3], dtype=np.int32),
        t_us=np.array([0, 50, 75, 100], dtype=np.int64),
        polarity=np.array([1, -1, 1, -1], dtype=np.int8),
        width=5,
        height=5,
        sequence_id="unit",
        t_start_us=0,
        t_end_us=100,
    )


def test_event_count_channels() -> None:
    encoded = encode_event_count(_events(), log1p=False)

    assert encoded.shape == (2, 5, 5)
    assert encoded[0].sum() == 2
    assert encoded[1].sum() == 2
    assert encoded[0, 1, 1] == 1
    assert encoded[1, 1, 1] == 1


def test_time_surface_recency() -> None:
    encoded = encode_time_surface(_events(), tau_ms=1.0)

    assert encoded.shape == (2, 5, 5)
    assert encoded[1, 3, 3] == 1.0
    assert encoded[0, 1, 1] < 1.0


def test_voxel_grid_preserves_event_weight_without_normalization() -> None:
    encoded = encode_voxel_grid(_events(), bins=3, normalize=False)

    assert encoded.shape == (6, 5, 5)
    assert encoded.sum() == np.float32(4.0)


def test_voxel_grid_matches_reference_histogram() -> None:
    rng = np.random.default_rng(7)
    width, height, bins = 11, 7, 4
    count = 128
    t_us = np.sort(rng.integers(10, 991, size=count, dtype=np.int64))
    events = EventBatch(
        x=rng.integers(0, width, size=count, dtype=np.int32),
        y=rng.integers(0, height, size=count, dtype=np.int32),
        t_us=t_us,
        polarity=rng.choice(np.asarray([-1, 1], dtype=np.int8), size=count),
        width=width,
        height=height,
        sequence_id="reference",
        t_start_us=10,
        t_end_us=991,
    )
    reference = np.zeros((bins * 2, height, width), dtype=np.float32)
    scaled = (t_us - 10) / 981.0 * (bins - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.ceil(scaled).astype(np.int64)
    upper_weight = scaled - lower
    for index in range(count):
        offset = 0 if events.polarity[index] > 0 else bins
        value = 1.0
        reference[offset + lower[index], events.y[index], events.x[index]] += value * (
            1.0 - upper_weight[index]
        )
        reference[offset + upper[index], events.y[index], events.x[index]] += (
            value * (upper_weight[index])
        )

    encoded = encode_voxel_grid(events, bins=bins, normalize=False)
    assert np.allclose(encoded, reference, atol=1e-6)


def test_voxel_grid_robust_normalization_preserves_empty_voxels() -> None:
    encoded = encode_voxel_grid(_events(), bins=3, normalize=True)
    occupied = encode_voxel_grid(_events(), bins=3, normalize=False) != 0

    assert np.all(encoded[~occupied] == 0.0)
    assert np.any(encoded[occupied] != 0.0)


def test_voxel_grid_robust_normalization_keeps_equal_occupied_voxels() -> None:
    events = EventBatch(
        x=np.array([1, 2], dtype=np.int32),
        y=np.array([1, 2], dtype=np.int32),
        t_us=np.array([0, 100], dtype=np.int64),
        polarity=np.array([1, -1], dtype=np.int8),
        width=4,
        height=4,
        sequence_id="equal",
        t_start_us=0,
        t_end_us=100,
    )

    encoded = encode_voxel_grid(events, bins=2, normalize=True)

    assert encoded[0, 1, 1] == 1.0
    assert encoded[3, 2, 2] == 1.0
    assert np.count_nonzero(encoded) == 2


def test_sparse_tokens_shape_and_bounds() -> None:
    tokens = encode_sparse_tokens(_events(), max_tokens=3, seed=1)

    assert tokens.shape == (3, 6)
    assert np.all(tokens[:, :3] >= 0)
    assert np.all(tokens[:, :3] <= 1)


def test_sparse_tokens_use_local_event_density() -> None:
    tokens = encode_sparse_tokens(_events(), max_tokens=4)

    assert tokens[:, 4].max() > tokens[:, 4].min()
    assert np.all(tokens[:, 4] > 0)
