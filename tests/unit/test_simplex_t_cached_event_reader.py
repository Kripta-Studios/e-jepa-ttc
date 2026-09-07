"""Exact chunk boundaries, dtype and lifecycle parity for retained handles."""

import h5py
import numpy as np
import pytest

from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader, ReaderPool


def fixture_file(path):
    t = np.repeat(np.arange(0, 10000, 17, dtype=np.int64), 2)
    with h5py.File(path, "w") as f:
        for key, values in {
            "t": t,
            "x": (t % 319).astype(np.int16),
            "y": (t % 239).astype(np.int16),
            "p": (t % 2).astype(np.int8),
        }.items():
            f.create_dataset(f"events/{key}", data=values, chunks=(40,), compression="gzip")
        f.create_dataset("ms_to_idx", data=np.searchsorted(t, np.arange(11) * 1000))


@pytest.mark.parametrize("bounds", [(0, 1), (17, 1000), (999, 4001), (1, 9999), (10000, 12000)])
@pytest.mark.parametrize("chunk", [1, 39, 40, 41, 250000])
def test_exact_reader(tmp_path, bounds, chunk):
    path = tmp_path / "events.h5"
    fixture_file(path)
    with EAPEventReader(path) as old, CachedEventReader(path) as new:
        a = list(old.iter_window_chunks(*bounds, chunk_events=chunk))
        b = list(new.iter_window_chunks(*bounds, chunk_events=chunk))
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            for key in left:
                assert left[key].dtype == right[key].dtype
                np.testing.assert_array_equal(left[key], right[key])
    assert not new.datasets and new._handle is None


def test_pool_reuses_and_closes(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    fixture_file(a)
    fixture_file(b)
    pool = ReaderPool()
    first = pool.get(a)
    assert pool.get(a) is first
    pool.get(b)
    assert first._handle is None
    pool.close()
    assert pool.reader is None


@pytest.mark.parametrize("bounds,chunk", [((-1, 2), 1), ((2, 2), 1), ((0, 2), 0)])
def test_invalid_window(tmp_path, bounds, chunk):
    path = tmp_path / "events.h5"
    fixture_file(path)
    with CachedEventReader(path) as reader, pytest.raises(ValueError):
        list(reader.iter_window_chunks(*bounds, chunk_events=chunk))
