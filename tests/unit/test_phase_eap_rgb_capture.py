"""Original-only eAP admission keeps deterministic sampling and strict parity."""

import numpy as np
import pytest

from scripts.capture_phase_eap_rgb_pair import parity, select_observations


def inventory():
    queries = [
        {
            "sample_token": f"{sequence}{index}",
            "sequence_id": sequence,
            "query_time_us": index,
            "rgb_history": [f"{sequence}{i}" for i in range(index + 1)],
        }
        for sequence in ("a", "b")
        for index in range(3)
    ]
    return {
        "queries": list(reversed(queries)),
        "physical_aliases": {q["sample_token"]: q["sample_token"] for q in queries},
    }


def test_pilot_selects_sequence_endpoints_independent_of_input_order():
    data = inventory()
    assert select_observations(data, "pilot") == ["a0", "a2", "b0", "b2"]
    data["queries"].reverse()
    assert select_observations(data, "pilot") == ["a0", "a2", "b0", "b2"]


def test_full_selects_all_histories_not_only_pilot_endpoints():
    assert select_observations(inventory(), "full") == ["a0", "a1", "a2", "b0", "b1", "b2"]


def test_duplicate_physical_query_is_rejected():
    data = inventory()
    data["queries"].append(data["queries"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        select_observations(data, "pilot")


def test_alias_is_not_an_extra_physical_query():
    data = inventory()
    data["physical_aliases"]["alias"] = "a0"
    data["queries"].append({**data["queries"][0], "sample_token": "alias"})
    assert select_observations(data, "full") == ["a0", "a1", "a2", "b0", "b1", "b2"]


def test_parity_retains_mismatch_and_distinguishes_exact():
    original = np.array([[1.0, 0.0]], dtype=np.float32)
    assert parity(original, original)["exact"]
    close = original + np.float32(1e-7)
    assert parity(close, original)["pass"]
    assert not parity(close, original)["exact"]
    wrong = original + np.float32(1e-3)
    assert not parity(wrong, original)["pass"]
    assert parity(wrong, original)["max_abs"] > 0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_original_is_never_admitted(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        parity(np.array([value], dtype=np.float32), np.array([1], dtype=np.float32))


def test_dtype_and_shape_changes_are_not_hidden_by_broadcasting():
    with pytest.raises(ValueError, match="shape/dtype"):
        parity(np.zeros((1, 3), np.float32), np.zeros(3, np.float32))
    with pytest.raises(ValueError, match="shape/dtype"):
        parity(np.zeros(3, np.float64), np.zeros(3, np.float32))
