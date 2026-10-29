"""History metadata normalization preserves decoding and conflict detection."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from e_jepa_ttc.rgb_port.data import decode_query
from scripts.runpod_rgb_metadata_v4 import install_history_adapter, metadata_values


def observation() -> dict:
    return {
        "observation_id": "same",
        "producer_shards": np.array(["s", "s"]),
        "producer_members": np.array(["a", "b"]),
        "producer_times_us": np.array([100_000, 200_000], dtype=np.int64),
        "producer_sensor_times_us": np.array([100_000, 200_000], dtype=np.int64),
        "producer_available_us": np.array([100_000, 200_000], dtype=np.int64),
        "producer_boxes_xyxy": np.array([[2, 2, 8, 8], [1, 1, 9, 9]], dtype=np.float32),
        "roi_xyxy": np.array([0, 0, 12, 12], dtype=np.float32),
        "anchor_us": np.int64(200_000),
    }


def test_duplicate_equal_metadata_is_scalar_and_conflicts_still_reject() -> None:
    first, second = observation(), observation()
    assert metadata_values(first) == metadata_values(second)
    second["producer_boxes_xyxy"][1, 0] += 1
    assert metadata_values(first) != metadata_values(second)
    assert isinstance(first["producer_boxes_xyxy"], np.ndarray)


def test_decoder_pixels_times_masks_exact_after_metadata_adapter() -> None:
    class Reader:
        def read_uint8(self, shard, member):
            return np.full((12, 12, 3), 17 if member == "a" else 29, dtype=np.uint8)

    row = observation()
    original = decode_query(row, eap_root=Path("."), reader=Reader())
    normalized = decode_query(metadata_values(row), eap_root=Path("."), reader=Reader())
    assert original.keys() == normalized.keys()
    for name in original:
        assert np.array_equal(original[name], normalized[name]), name


def test_only_history_is_adapted_and_source_values_are_not_modified() -> None:
    class Source:
        def history(self, index):
            return [observation()]

        def row(self, index):
            return {"target": np.float32(7)}

    module = SimpleNamespace(AllTrainRGBSource=Source)
    install_history_adapter(module)
    assert Source().history(0) == Source().history(1)
    assert isinstance(Source().row(0)["target"], np.float32)


def test_shape_value_preservation_and_nonfinite_nested_metadata_rejected() -> None:
    array = np.array([[1.25, 2.5], [3.75, 4.125]], dtype=np.float32)
    rebuilt = np.asarray(metadata_values(array), dtype=array.dtype)
    assert rebuilt.shape == array.shape and rebuilt.dtype == array.dtype
    assert rebuilt.tobytes() == array.tobytes()
    for invalid in (float("nan"), float("inf"), np.float32("nan")):
        with pytest.raises(ValueError, match="Nonfinite"):
            metadata_values({"nested": [np.array([invalid])]})
