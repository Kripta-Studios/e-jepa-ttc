"""Pure CPU lookahead contracts; no Torch or dataset imports are required."""

import random
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from operational.rgb_port_rgb_overlap.prefetch import (
    EffectiveRGBPrefetch,
    capture_effective_prefetch,
)


class RawSource:
    population_size = 9
    frame_counts = [2, 3, 2, 3, 3, 3, 2, 3, 3]
    identity = {"fixed": True}

    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def batch(self, ids, modality):
        assert modality == "rgb"
        self.calls.append(list(ids))
        if self.fail in ids:
            raise OSError("bounded retry exhausted")
        return np.stack([np.full((self.frame_counts[index], 2), index) for index in ids])


class Effective:
    def __init__(self, source, recipe=None, revision=None):
        self.source = source
        self.identity = source.identity
        self.population_size = source.population_size
        self.frame_counts = source.frame_counts

    def batch(self, ids, modality):
        groups = {}
        for index in ids:
            groups.setdefault(self.frame_counts[index], []).append(index)
        return SimpleNamespace(
            parts=[self.source.batch(groups[key], modality) for key in sorted(groups)]
        )


def test_mixed_parts_tail_epoch_cursor_rng_and_storage_are_preserved():
    raw = RawSource()
    order = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4])
    cursor = {"order": order, "position": 0}
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    original_init = Effective.__init__
    with capture_effective_prefetch(Effective, expected_source=raw, batch_size=4) as capture:
        effective = Effective(raw)
        assert isinstance(effective, Effective) and effective.identity is raw.identity
        capture.bind(cursor)
        capture.bind(cursor)
        assert capture.captured and capture.bound and cursor["position"] == 0
        first = effective.batch(order[:4].tolist(), "rgb")
        preserved = [part.copy() for part in first.parts]
        assert [part.shape[1] for part in first.parts] == [2, 3]
        cursor["position"] = 4
        effective.batch(order[4:8].tolist(), "rgb")
        cursor["position"] = 8
        tail = effective.batch([4], "rgb")
        assert tail.parts[0].shape == (1, 3, 2)
        assert capture.snapshot()["submitted_batches"] == 3
        with pytest.raises(RuntimeError, match="new canonical order"):
            effective.batch([4], "rgb")
        cursor.update(order=np.arange(9), position=0)
        effective.batch([0, 1, 2, 3], "rgb")
        assert cursor["position"] == 0
        for before, after in zip(preserved, first.parts, strict=True):
            np.testing.assert_array_equal(before, after)
        assert capture.snapshot()["queued_batches_max"] == 2
    assert "batch" not in vars(effective) and Effective.__init__ is original_init
    assert random.getstate() == random_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])


def test_synchronous_before_bind_and_capture_scoped_to_expected_source():
    raw = RawSource()
    with capture_effective_prefetch(Effective, expected_source=raw, batch_size=4) as capture:
        other = Effective(RawSource())
        assert not capture.captured and "batch" not in vars(other)
        with pytest.raises(RuntimeError, match="not been captured"):
            capture.bind({"order": np.arange(9), "position": 0})
        source = Effective(raw)
        source.batch([0, 1], "rgb")
        assert not capture.bound and capture.snapshot()["submitted_batches"] == 0
        assert raw.calls == [[0], [1]]
        with pytest.raises(RuntimeError, match="more than once"):
            Effective(raw)


def test_future_errors_are_deferred_retained_and_wrong_ids_do_not_consume():
    loader = EffectiveRGBPrefetch(Effective(RawSource(fail=4)), batch_size=4)
    cursor = {"order": np.arange(9), "position": 0}
    try:
        loader.bind(cursor)
        with pytest.raises(RuntimeError, match="replace canonical cursor"):
            loader.bind(dict(cursor))
        with pytest.raises(RuntimeError, match="canonical order"):
            loader.batch([3, 2, 1, 0], "rgb")
        loader.batch([0, 1, 2, 3], "rgb")
        with pytest.raises(RuntimeError, match="canonical order"):
            loader.batch([4, 5, 6, 7], "rgb")
        cursor["position"] = 4
        for _ in range(2):
            with pytest.raises(OSError, match="bounded retry exhausted"):
                loader.batch([4, 5, 6, 7], "rgb")
        assert loader.snapshot()["consumed_batches"] == 1
    finally:
        loader.close()


def test_prefetch_is_asynchronous_and_close_joins_before_restoring_method():
    raw = RawSource()
    entered, release, closed = Event(), Event(), Event()

    class Blocking(Effective):
        def batch(self, ids, modality):
            entered.set()
            assert release.wait(5), "test release was not delivered"
            return super().batch(ids, modality)

    with capture_effective_prefetch(Blocking, expected_source=raw, batch_size=4) as capture:
        source = Blocking(raw)
        capture.bind({"order": np.arange(9), "position": 0})
        assert entered.wait(2)
        # bind returned while input preparation is still blocked.
        assert capture.snapshot()["consumed_batches"] == 0

        def close():
            capture.close()
            closed.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(close)
            assert not closed.wait(0.05)
            assert "batch" in vars(source)
            release.set()
            future.result(timeout=5)
        assert closed.is_set() and "batch" not in vars(source)


def test_constructor_override_is_restored_and_failure_closes_capture():
    raw = RawSource()

    class Override(Effective):
        def __init__(self, source):
            super().__init__(source)
            self.batch = lambda ids, modality: (ids, modality)

    source = None
    with pytest.raises(ValueError, match="sentinel"):
        with capture_effective_prefetch(Override, expected_source=raw, batch_size=4) as capture:
            source = Override(raw)
            assert capture.loader is not None
            original = capture.loader.original_batch
            raise ValueError("sentinel")
    assert source is not None and source.batch is original


@pytest.mark.parametrize("position", [-1, 10])
def test_reject_invalid_cursor_position(position):
    loader = EffectiveRGBPrefetch(Effective(RawSource()), batch_size=4)
    try:
        with pytest.raises(ValueError, match="canonical RGB order"):
            loader.bind({"order": np.arange(9), "position": position})
    finally:
        loader.close()


def test_mid_epoch_restore_and_close_are_final():
    loader = EffectiveRGBPrefetch(Effective(RawSource()), batch_size=4)
    loader.bind({"order": np.arange(9), "position": 8})
    loader.batch([8], "rgb")
    assert loader.snapshot()["submitted_batches"] == 1
    loader.close()
    loader.close()
    with pytest.raises(ValueError, match="Open RGB"):
        loader.batch([8], "rgb")
    with pytest.raises(RuntimeError, match="closed"):
        loader.bind({"order": np.arange(9), "position": 8})
