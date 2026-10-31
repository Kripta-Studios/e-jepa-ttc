import numpy as np

from operational.phase_robust.cache import SensorCache


class Source:
    def __init__(self):
        self.calls = 0
        self.events = np.column_stack(
            (np.zeros(100), np.zeros(100), np.arange(100) * 10000, np.ones(100))
        ).astype(np.int64)

    def read_events(self, start, end):
        self.calls += 1
        return self.events[(self.events[:, 2] >= start) & (self.events[:, 2] < end)].copy()

    def read_rgb(self, index):
        return np.full((2, 3, 3), index, dtype=np.uint8)


def test_cache_half_open_boundaries_exact_parity_hits_and_bounded_memory():
    source = Source()
    cached = SensorCache(source, max_event_bytes=2000)
    for start, end in [
        (0, 200000),
        (10000, 199999),
        (99999, 200001),
        (600000, 800000),
        (0, 200000),
    ]:
        expected = source.events[(source.events[:, 2] >= start) & (source.events[:, 2] < end)]
        np.testing.assert_array_equal(cached.read_events(start, end), expected)
        assert cached.event_bytes <= 2000
    assert cached.hits > 0
    image = cached.read_rgb(2)
    image[:] = 0
    assert cached.read_rgb(2).min() == 2
