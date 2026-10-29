from __future__ import annotations

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.garlttc_lhr_cache import (
    OBSERVABLE_MOTION_DIM,
    GarlTTCLHRCacheConfig,
    _atomic_torch_save,
    _box_features,
    _load_torch_records,
    observable_motion_from_boxes_torch,
)


def test_observable_motion_has_no_ttc_or_depth_dependency() -> None:
    first = (400.0, 200.0, 500.0, 400.0)
    second = (430.0, 200.0, 530.0, 410.0)
    values, metadata = _box_features(first, second, 0.1)
    assert values.shape == (OBSERVABLE_MOTION_DIM,)
    assert np.isfinite(values).all()
    assert metadata["lateral_speed_raw"] > 0.0


def test_torch_observable_motion_accepts_normalized_boxes() -> None:
    boxes = torch.tensor(
        [
            [
                [0.30, 0.30, 0.40, 0.60],
                [0.32, 0.30, 0.42, 0.61],
            ]
        ],
        dtype=torch.float32,
    )
    motion = observable_motion_from_boxes_torch(boxes, torch.tensor([0.1]))
    assert motion.shape == (1, OBSERVABLE_MOTION_DIM)
    assert torch.isfinite(motion).all()


def test_gzip_shard_round_trip_is_lossless(tmp_path) -> None:
    path = tmp_path / "shard-00000.pt.gz"
    records = [
        {"tensor": np.arange(12, dtype=np.float32).reshape(3, 4), "label": np.int64(-2)},
        {"tensor": np.ones((2, 2), dtype=np.float32), "label": np.int64(7)},
    ]
    _atomic_torch_save(records, path, compression="gzip", compression_level=1)
    loaded = _load_torch_records(path)
    assert len(loaded) == len(records)
    for actual, expected in zip(loaded, records, strict=True):
        assert actual["label"] == expected["label"]
        np.testing.assert_array_equal(actual["tensor"], expected["tensor"])


def test_cache_config_rejects_unknown_compression() -> None:
    with pytest.raises(ValueError, match="compression"):
        GarlTTCLHRCacheConfig(compression="lz4")
