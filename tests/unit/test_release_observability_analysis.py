"""Ensure group probes never fit scalers or coefficients on the held-out acquisition."""

import numpy as np
import pytest

from scripts.release_observability_analysis import (
    balanced_weights,
    design,
    grouped_probe,
    permute_physical,
    summarize_probes,
)


def test_scalers_and_imputation_exclude_validation_values():
    train = np.ones((4, 6))
    train[:, 0] = np.nan
    validation = np.ones((2, 6)) * 1000
    _, transformed, receipt = design(train, validation)
    assert receipt["medians"][0] == 0
    assert receipt["mean"][1] == 1
    assert transformed[0, 2] == pytest.approx(999e6)


def test_group_weights_and_predictions_do_not_leak_held_out_targets():
    groups = np.array(["a"] * 3 + ["b"] * 2 + ["c"] * 5)
    weights = balanced_weights(groups)
    assert weights[groups == "a"].sum() == pytest.approx(weights[groups == "c"].sum())
    x = np.arange(60, dtype=float).reshape(10, 6)
    y = np.linspace(-1.0, 1.0, 10)
    old, fits = grouped_probe(x, y, groups)
    changed = y.copy()
    changed[groups == "b"] += 10000
    new, _ = grouped_probe(x, changed, groups)
    np.testing.assert_array_equal(old[groups == "b"], new[groups == "b"])
    assert all(f["held_out_group"] not in f["training_groups"] for f in fits)


def test_permutation_keeps_missingness_and_is_reproducible():
    x = np.arange(240, dtype=float).reshape(40, 6)
    x[::2, 2] = np.nan
    support = np.zeros_like(x)
    groups = np.array(["road"] * 40)
    shuffled, receipt = permute_physical(x, support, groups)
    again, second = permute_physical(x, support, groups)
    np.testing.assert_array_equal(np.isfinite(x), np.isfinite(shuffled))
    np.testing.assert_array_equal(shuffled, again)
    assert receipt == second and receipt["changed_fraction"] > 0.25


def test_probe_admission_rejects_physics_no_better_than_control():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.arange(6, dtype=float)
    predictions = {name: np.zeros(6) for name in ("constant", "support", "physical", "permuted")}
    summary = summarize_probes(y, predictions, groups, {"changed_fraction": 1.0})
    assert not summary["gate_pilot_admitted"]
    predictions["physical"] = y.copy()
    assert summarize_probes(y, predictions, groups, {"changed_fraction": 1.0})[
        "gate_pilot_admitted"
    ]
    assert not summarize_probes(y, predictions, groups, {"changed_fraction": 0.1})[
        "gate_pilot_admitted"
    ]
