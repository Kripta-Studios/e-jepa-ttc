"""Real HDF5 endpoint checks, with no expert inference or target access."""

import json

import h5py
import numpy as np
import pytest

from e_jepa_ttc.simplex_t.expanded_stream_support import (
    verify_expanded_stream_support,
    verify_stream_receipts,
)


@pytest.fixture
def support(tmp_path):
    directory = tmp_path / "train" / "seq"
    directory.mkdir(parents=True)
    path = directory / "events.h5"
    with h5py.File(path, "w") as stream:
        stream.create_dataset("events/t", data=np.array([0, 400_000, 900_000], dtype=np.int64))
    lag = np.arange(15, -1, -1, dtype=np.int64) * 50_000
    windows = np.array([[[300_000, 400_000], [400_000, 500_000], [500_000, 600_000]]])
    index = {
        "sequences": np.array(["seq"]),
        "lag_us": lag,
        "base_windows_us": windows,
        "anchor_us": np.array([600_000], dtype=np.int64),
        "valid": (300_000 - lag >= 0)[None, :],
    }
    args = {
        "pool": "D1",
        "raw_train_root": tmp_path / "train",
        "allowed_sequences": {"seq"},
        "resource_ok": lambda: True,
    }
    return index, {"stream_bounds_us": {"seq": [0, 900_000]}}, args, path


@pytest.mark.parametrize("pool", ["D1", "DENSE_OLD"])
def test_actual_endpoints_recompute_masks_without_targets(support, pool):
    index, manifest, args, _ = support
    args["pool"] = pool
    manifest["stream_bounds_us"]["seq"][1] += int(pool == "DENSE_OLD")
    receipts = verify_expanded_stream_support(index, manifest, **args)
    assert receipts[0]["valid_slots"] == 7
    assert receipts[0]["queries"] == 1
    assert receipts[0]["events"] == 3
    assert json.loads(json.dumps(receipts)) == receipts
    verify_stream_receipts(receipts)


@pytest.mark.parametrize("mutation", ["mask", "bounds", "anchor", "lag", "roles", "dtype"])
def test_support_rejects_index_disagreement(support, mutation):
    index, manifest, args, _ = support
    if mutation == "mask":
        index["valid"][0, 0] = True
    elif mutation == "bounds":
        manifest["stream_bounds_us"]["seq"][1] += 1
    elif mutation == "anchor":
        index["anchor_us"][0] += 1
    elif mutation == "lag":
        index["lag_us"][0] += 1
    elif mutation == "roles":
        args["allowed_sequences"] = {"other"}
    else:
        index["base_windows_us"] = index["base_windows_us"].astype(float)
    with pytest.raises(ValueError):
        verify_expanded_stream_support(index, manifest, **args)


def test_resource_pause_before_raw_open(support, monkeypatch):
    index, manifest, args, _ = support
    args["resource_ok"] = lambda: False

    def forbidden(*args, **kwargs):
        raise AssertionError("raw file opened without resources")

    monkeypatch.setattr(h5py, "File", forbidden)
    with pytest.raises(InterruptedError, match="PAUSED_RESOURCE"):
        verify_expanded_stream_support(index, manifest, **args)


def test_source_mutation_rejected_at_later_boundary(support):
    index, manifest, args, path = support
    receipts = verify_expanded_stream_support(index, manifest, **args)
    with h5py.File(path, "a") as stream:
        stream.create_dataset("extra", data=np.arange(100))
    with pytest.raises(ValueError, match="changed after"):
        verify_stream_receipts(receipts)


@pytest.mark.parametrize("values", [[], [100, 0], [-1, 900_000]])
def test_invalid_raw_endpoints_rejected(support, values):
    index, manifest, args, path = support
    with h5py.File(path, "w") as stream:
        stream.create_dataset("events/t", data=np.array(values, dtype=np.int64))
    with pytest.raises(ValueError):
        verify_expanded_stream_support(index, manifest, **args)


def test_timestamp_group_is_not_a_dataset(support):
    index, manifest, args, path = support
    with h5py.File(path, "w") as stream:
        stream.create_group("events/t")
    with pytest.raises(ValueError, match="integer timestamp stream"):
        verify_expanded_stream_support(index, manifest, **args)
