"""Ensure a feasibility timing pilot cannot silently include forbidden folds or repeats."""

import numpy as np
import pytest

from operational.efficient_context.train40_pilot import digest_array, select_queries


def fixture_index() -> tuple[dict, list[dict]]:
    index = {
        "sequences": np.array(["train_a", "train_a", "train_b", "old", "wrong"]),
        "tokens": np.array(["a0", "a1", "b0", "old0", "wrong0"]),
        "producer_family": np.array([[0, 0, 1, 2, 3]]),
        "valid": np.ones((5, 16), dtype=bool),
    }
    families = [
        {"outer_fold": 0, "role": "inner0", "experts": {}},
        {"outer_fold": 0, "role": "inner1", "experts": {}},
        {"outer_fold": 0, "role": "outer_dev", "experts": {}},
        {"outer_fold": 1, "role": "inner0", "experts": {}},
    ]
    return index, families


def test_pilot_excludes_old_dev_and_wrong_outer() -> None:
    index, families = fixture_index()
    rows = select_queries(index, families, set())
    assert {r["sample_token"] for r in rows} == {"a0", "a1", "b0"}


def test_pilot_excludes_previous_timing_requests() -> None:
    index, families = fixture_index()
    rows = select_queries(index, families, {"a0", "b0"})
    assert [r["sample_token"] for r in rows] == ["a1"]


def test_pilot_rejects_absent_current_observations() -> None:
    index, families = fixture_index()
    index["valid"][:, -1] = False
    with pytest.raises(ValueError, match="nonempty"):
        select_queries(index, families, set())


def test_pilot_selection_never_requires_targets() -> None:
    index, families = fixture_index()
    first = select_queries(index, families, set())
    index["ttc_labels_do_not_read"] = object()
    assert select_queries(index, families, set()) == first


def test_pilot_full_tensor_hash_detects_one_changed_voxel() -> None:
    array = np.zeros((16, 3, 12, 4, 4), dtype=np.float32)
    original = digest_array(array)
    array[-1, -1, -1, -1, -1] = np.nextafter(np.float32(0), np.float32(1))
    assert digest_array(array) != original
