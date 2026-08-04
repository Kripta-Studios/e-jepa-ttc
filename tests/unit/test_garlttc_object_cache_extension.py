from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from e_jepa_ttc.data.garlttc_lhr_cache import (
    GarlTTCLHRCacheConfig,
    _MaskReader,
    _atomic_torch_save,
    _jepa_roi_voxel,
    _load_torch_records,
)


def test_jepa_roi_voxel_is_object_specific_and_21_channel() -> None:
    events = {
        "x": np.asarray([10, 20, 90], dtype=np.int32),
        "y": np.asarray([10, 20, 90], dtype=np.int32),
        "t": np.asarray([0, 50_000, 99_999], dtype=np.int64),
        "p": np.asarray([1, -1, 1], dtype=np.int8),
    }
    kwargs = {
        "square": (0.0, 0.0, 40.0, 40.0),
        "size": 32,
        "bins": 5,
        "sequence_id": "synthetic",
        "start_us": 0,
        "end_us": 100_000,
        "event_pixel_diff": 0,
    }
    tensor = _jepa_roi_voxel(events, **kwargs)
    without_outside = {key: value[:2] for key, value in events.items()}
    reference = _jepa_roi_voxel(without_outside, **kwargs)
    assert tensor.shape == (21, 32, 32)
    assert torch.isfinite(tensor).all()
    assert float(tensor.abs().sum()) > 0.0
    # The event at (90, 90) lies outside the target ROI and must not contribute.
    torch.testing.assert_close(tensor, reference)


def test_mask_reader_never_invents_a_box_mask(tmp_path: Path) -> None:
    reader = _MaskReader(tmp_path)
    missing, valid = reader.read(
        "does-not-exist.png",
        square=(0.0, 0.0, 16.0, 16.0),
        size=8,
    )
    assert not valid
    assert missing.sum().item() == 0.0

    raw = np.zeros((16, 16), dtype=np.uint8)
    raw[4:8, 5:9] = 255
    Image.fromarray(raw).save(tmp_path / "mask.png")
    mask, valid = reader.read(
        "mask.png",
        square=(0.0, 0.0, 16.0, 16.0),
        size=8,
    )
    assert valid
    assert 0 < mask.sum().item() < 64


def test_object_lhr_minimal_storage_configuration_is_valid() -> None:
    config = GarlTTCLHRCacheConfig(
        shard_size=16,
        store_full_frame_events=False,
        store_garl_event_roi=False,
        store_jepa_event_roi=True,
    )
    assert not config.store_full_frame_events
    assert not config.store_garl_event_roi
    assert config.store_jepa_event_roi


def test_atomic_torch_shard_roundtrip(tmp_path: Path) -> None:
    records = [{"jepa_event_roi": np.zeros((2, 21, 8, 8), dtype=np.float32)}]
    path = tmp_path / "shard.pt"
    _atomic_torch_save(records, path)
    loaded = _load_torch_records(path)
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0]["jepa_event_roi"], records[0]["jepa_event_roi"])
