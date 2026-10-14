"""Native miss/hit decodes must remain exact after dropping unused T3/mask storage."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from e_jepa_ttc.rgb_port.data import PreparedRGBCache
from operational.joint_measurement.rgb_p2_cached_data import (
    PAIR_FIELDS,
    STATIC_FIELDS,
    P2RGBCache,
    compact_shard,
)
from operational.range_transfer.common import sha


@pytest.mark.parametrize("steps", [2, 3])
def test_compaction_native_hash_decode_and_hits_preserve_every_consumed_value(tmp_path, steps):
    rng = np.random.default_rng(7)
    arrays = {
        "indices": np.arange(2, dtype=np.int64),
        "tokens": np.array(["q0", "q1"]),
        "rgb_uint8": rng.integers(0, 256, (2, steps, 3, 128, 128), dtype=np.uint8),
        "sensor_frame_times_us": np.tile(np.arange(steps, dtype=np.int64) * 100001, (2, 1)),
        "annotation_frame_times_us": np.tile(np.arange(steps, dtype=np.int64) * 100000, (2, 1)),
        "delta_t_s": np.full((2, steps - 1), np.float32(0.100001)),
        "boxes_in_crop_xyxy": np.ones((2, steps, 4), np.float32),
        "visible_heights_px": np.full((2, steps), 17.5, np.float32),
        "roi_xyxy": np.ones((2, 4), np.int32),
        "foreground_mask": np.ones((2, steps, 1, 128, 128), np.bool_),
    }
    compact = compact_shard(arrays)
    assert not np.shares_memory(compact["rgb_uint8"], arrays["rgb_uint8"])
    shard, binding, manifest = (tmp_path / name for name in ("input.npz", "binding", "manifest"))
    np.savez_compressed(shard, **arrays)
    binding.write_text("binding")
    manifest.write_text("manifest")
    entry = {"path": str(shard), "sha256": sha(shard), "tokens": ["q0", "q1"]}
    base = SimpleNamespace(
        binding_path=binding,
        manifest_path=manifest,
        _lookup={i: (entry, i) for i in range(2)},
        _loaded=OrderedDict(),
        _loaded_bytes=0,
    )
    base.get = lambda index, token: PreparedRGBCache.get(base, index, token)
    reader = P2RGBCache(base, limit_bytes=256 * 1024**2)
    for i in (0, 1, 0):
        sample = reader.get(i, f"q{i}")
        np.testing.assert_array_equal(
            sample["rgb"][-2:], arrays["rgb_uint8"][i, -2:].astype(np.float32) / np.float32(255)
        )
        for field in PAIR_FIELDS:
            np.testing.assert_array_equal(sample[field][-2:], arrays[field][i, -2:])
        np.testing.assert_array_equal(sample["delta_t_s"][-1:], arrays["delta_t_s"][i, -1:])
        np.testing.assert_array_equal(sample["roi_xyxy"], arrays["roi_xyxy"][i])
    resident = base._loaded[str(shard)]
    assert set(resident) == {*PAIR_FIELDS, *STATIC_FIELDS, "delta_t_s"}
    assert base._loaded_bytes == sum(value.nbytes for value in resident.values())
    assert reader.compactions == 1 and reader.hits == 2 and reader.misses == 1
    assert reader.discarded_bytes > 0
    shard.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="shard changed"):
        reader.get(0, "q0")
