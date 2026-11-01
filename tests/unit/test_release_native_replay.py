"""Historical parity tolerances and missing-field behavior are not learned from test outputs."""

import numpy as np
import pytest

from scripts.release_native_pair_replay import compare_arrays, query_observations


def test_historical_parity_requires_complete_fields_and_exact_categories():
    expected = {"mask": np.array([True, False]), "raw": np.array([0.1], np.float32)}
    assert all(v["passed"] for v in compare_arrays(expected, expected).values())
    changed = {**expected, "mask": np.array([True, True])}
    assert not compare_arrays(changed, expected)["mask"]["passed"]
    with pytest.raises(ValueError, match="inventory"):
        compare_arrays({"raw": expected["raw"]}, expected)
    with pytest.raises(ValueError, match="inventory"):
        compare_arrays({}, {})


def test_historical_tolerance_keeps_ttc_separate_from_raw_and_finite():
    expected = {"raw": np.array([1.0], np.float32), "head.output_ttc": np.array([1.0], np.float32)}
    actual = {k: v + 0.00005 for k, v in expected.items()}
    result = compare_arrays(actual, expected)
    assert not result["raw"]["passed"]
    assert result["head.output_ttc"]["passed"]
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_arrays({k: np.full_like(v, np.nan) for k, v in expected.items()}, expected)


def test_query_history_retains_order_and_omits_only_explicit_rejections():
    query = {"event_history": ["b", "a"], "rgb_history": ["c", "d"]}
    cache = {"event": {"a": {"x": 1}, "b": {"x": 2}}, "rgb": {"c": None, "d": {"x": 4}}}
    assert query_observations(query, cache) == {"event": [{"x": 2}, {"x": 1}], "rgb": [{"x": 4}]}
    with pytest.raises(KeyError):
        query_observations({"event_history": ["missing"], "rgb_history": []}, cache)
