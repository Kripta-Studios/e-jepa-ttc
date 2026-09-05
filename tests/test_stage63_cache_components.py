"""Adversarial checks for complete cache component identity."""

from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.artifacts.cache_components import component_record, verify_components
from e_jepa_ttc.data.raw_temporal_cache import convert_counts_bounded


def test_bounded_count_conversion_preserves_source_and_interrupted_attempt(tmp_path: Path) -> None:
    original = np.arange(137 * 24, dtype=np.uint32).reshape(137, 2, 3, 4)
    source, target = tmp_path / "counts.npy", tmp_path / "converted.npy"
    np.save(source, original)
    source_record = component_record(source)
    checks = 0

    def interrupt() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise TimeoutError("simulated RAM margin")

    with pytest.raises(TimeoutError):
        convert_counts_bounded(source, target, "uint16", interrupt)
    prior_bytes = target.read_bytes()
    convert_counts_bounded(source, target, "uint16")
    assert np.array_equal(np.load(target), original)
    assert component_record(source) == source_record
    previous = list(tmp_path.glob("converted.npy.previous_*"))
    assert len(previous) == 1 and previous[0].read_bytes() == prior_bytes


def test_bounded_count_conversion_rejects_overflow(tmp_path: Path) -> None:
    source, target = tmp_path / "counts.npy", tmp_path / "converted.npy"
    np.save(source, np.full((2, 2), 65536, dtype=np.uint32))
    with pytest.raises(ValueError, match="overflow"):
        convert_counts_bounded(source, target, "uint16")


@pytest.mark.parametrize("mutation", ["value", "order", "dtype", "shape"])
def test_cache_rejects_numerical_or_layout_changes(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "state.npy"
    array = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.save(path, array)
    records = {path.name: component_record(path)}
    verify_components(tmp_path, records, {path.name})
    if mutation == "value":
        array[0, 0] = 99
    elif mutation == "order":
        array = array[::-1]
    elif mutation == "dtype":
        array = array.astype(np.float64)
    else:
        array = array.reshape(3, 4)
    np.save(path, array)
    with pytest.raises(ValueError, match="bytes/layout mismatch"):
        verify_components(tmp_path, records, {path.name})


def test_cache_rejects_incomplete_inventory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inventory"):
        verify_components(tmp_path, {}, {"durations.npy"})


def test_cache_rejects_parent_escape(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "state.npy"
    np.save(outside, np.ones(3))
    record = component_record(outside)
    record["path"] = "../state.npy"
    with pytest.raises(ValueError, match="escapes"):
        verify_components(root, {"../state.npy": record}, {"../state.npy"})
