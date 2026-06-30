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


def test_sparse_tokens_shape_and_bounds() -> None:
    tokens = encode_sparse_tokens(_events(), max_tokens=3, seed=1)

    assert tokens.shape == (3, 6)
    assert np.all(tokens[:, :3] >= 0)
    assert np.all(tokens[:, :3] <= 1)
