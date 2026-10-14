"""Resident RGB shards must detect mutation and release memory under pressure."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from operational.joint_measurement import rgb_cached_data as runtime


def cache(tmp_path):
    shard, binding, manifest = (
        tmp_path / name for name in ("shard.npz", "binding.json", "manifest.json")
    )
    for path in (shard, binding, manifest):
        path.write_bytes(b"first")
    base = SimpleNamespace(
        binding_path=binding,
        manifest_path=manifest,
        _lookup={0: ({"path": str(shard)}, 0)},
        _loaded=OrderedDict(),
        _loaded_bytes=0,
    )
    values = np.arange(8, dtype=np.uint8)

    def get(index, token):
        assert index == 0 and token == "q0"
        if str(shard) not in base._loaded:
            base._loaded[str(shard)] = {"rgb_uint8": values}
            base._loaded_bytes += values.nbytes
        return {"rgb": values.astype(np.float32) / np.float32(255)}

    base.get = get
    return runtime.GuardedRGBCache(base, limit_bytes=1024**3), shard, binding


def test_memory_hit_preserves_native_values_but_rejects_changed_disk_source(tmp_path):
    reader, shard, _ = cache(tmp_path)
    first = reader.get(0, "q0")
    second = reader.get(0, "q0")
    np.testing.assert_array_equal(first["rgb"], second["rgb"])
    assert reader.counters()["hits"] == reader.counters()["misses"] == 1
    shard.write_bytes(b"changed-source")
    with pytest.raises(ValueError, match="shard changed"):
        reader.get(0, "q0")


def test_binding_changes_fail_before_reusing_resident_rows(tmp_path):
    reader, _, binding = cache(tmp_path)
    reader.validate_binding()
    binding.write_bytes(b"different binding")
    with pytest.raises(ValueError, match="manifest or binding changed"):
        reader.validate_binding()


def test_memory_pressure_evicts_old_shards_before_loading(monkeypatch, tmp_path):
    reader, _, _ = cache(tmp_path)
    for name in ("old0", "old1"):
        reader.base._loaded[name] = {"fake_array": SimpleNamespace(nbytes=300 * 1024**2)}
    reader.base._loaded_bytes = 600 * 1024**2
    monkeypatch.setattr(
        runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=3 * 1024**3)
    )
    reader.get(0, "q0")
    assert "old0" not in reader.base._loaded and "old1" not in reader.base._loaded
    assert reader.counters()["retained_bytes"] == 8
