from dataclasses import replace

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.history import (
    Observation,
    build_history,
    feature_bytes,
    history_arrays,
    verify_ancestry,
)


def rows(n=20):
    return [
        Observation(str(i), "s", "t", "p", 10**16 + i * 100002, 10**16 + i * 100002 + 1000)
        for i in range(n)
    ]


@pytest.mark.parametrize("length", [1, 4, 8, 16])
def test_lengths(length):
    r = rows()
    q = np.array([1, 19])
    h = build_history(r, q, length)
    assert h.shape == (2, length) and np.all(h[:, -1] == q)
    assert np.all(h[1] == np.arange(20 - length, 20))


def test_future_availability_excluded():
    r = rows(5)
    r[3] = replace(r[3], available_us=r[4].available_us + 1)
    assert 3 not in build_history(r, np.array([4]), 8)[0]


def test_gap_resets():
    r = rows(5)
    r[4] = replace(r[4], anchor_us=r[3].anchor_us + 400000, available_us=r[3].available_us + 400000)
    h = build_history(r, np.array([4]), 8)
    assert np.sum(h >= 0) == 1


def test_track_isolation():
    r = rows(5)
    r[3] = replace(r[3], track="other")
    assert 3 not in build_history(r, np.array([4]), 8)[0]


def test_producer_isolation():
    r = rows(5)
    r[3] = replace(r[3], producer="other")
    assert 3 not in build_history(r, np.array([4]), 8)[0]


def test_duplicate_anchor_refused():
    r = rows(5)
    r[3] = replace(r[3], anchor_us=r[2].anchor_us)
    with pytest.raises(ValueError):
        build_history(r, np.array([4]), 8)


def test_protected_role():
    with pytest.raises(ValueError):
        Observation("a", "s", "t", "p", 0, 0, "TEST")


def test_float_timestamp_rejected():
    with pytest.raises(ValueError):
        Observation("a", "s", "t", "p", 1.0, 2)


def test_int_before_float():
    r = rows(5)
    h = build_history(r, np.array([4]), 8)
    x, t, v = history_arrays(r, np.ones((5, 17)), h)
    assert np.isclose(t[0, -1, 2], 0.100002) and t[0, -1, 0] == 0
    assert np.all(x[~v] == 0)


def test_no_future_anchor():
    r = rows(8)
    h = build_history(r, np.array([3]), 8)
    assert np.all(h <= 3)


def test_order_and_query_identity():
    h = build_history(rows(8), np.array([7, 2, 5]), 4)
    assert h[:, -1].tolist() == [7, 2, 5]


def test_empty_history_current_exists():
    h = build_history(rows(1), np.array([0]), 16)
    assert h[0, -1] == 0 and np.all(h[0, :-1] == -1)


@pytest.mark.parametrize("fit,obs,outer", [({"a"}, {"a"}, {"b"}), ({"b"}, {"a"}, {"b"})])
def test_ancestry(fit, obs, outer):
    with pytest.raises(ValueError):
        verify_ancestry(fit, obs, outer)


def test_valid_ancestry():
    verify_ancestry({"a"}, {"b"}, {"c"})


def test_memory_formula():
    sizes = feature_bytes(100000, 100000, 64)
    assert sizes["deduplicated_float32_features"] == 25600000
    assert sizes["int32_context_index"] == 6400000


def test_no_duplicate_queries():
    with pytest.raises(ValueError):
        build_history(rows(3), np.array([2, 2]))


def test_h1_does_not_read_previous_gap():
    r = rows(5)
    h = build_history(r, np.array([4]), 1)
    _, t, _ = history_arrays(r, np.ones((5, 17)), h)
    assert t[0, 0, 2] == 0


def test_future_dependency_adapter_rejected():
    r = rows(5)
    r[3] = replace(r[3], available_us=r[4].available_us + 1)
    with pytest.raises(ValueError):
        history_arrays(r, np.ones((5, 17)), np.array([[3, 4]]))
