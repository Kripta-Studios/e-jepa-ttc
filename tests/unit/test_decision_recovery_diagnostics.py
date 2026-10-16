"""Frozen diagnostic identity, conditional populations and non-deployable oracles."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from operational.decision_recovery import diagnostics


def table(predictions, role="P"):
    target = np.array([-2.0, -3.0, -4.0, -5.0, 1.0, 2.0, 4.0, 8.0])
    prediction = np.array(predictions, dtype=np.float64)
    return pd.DataFrame(
        {
            "sample_token": [f"query_{i}" for i in range(len(target))],
            "group_id": ["g1"] * 4 + ["g2"] * 4,
            "sequence_id": ["s1"] * 4 + ["s2"] * 4,
            "target_ttc": target,
            "prediction_ttc": prediction,
            "raw_phase": -np.log1p(-0.1 / prediction),
            "role": role,
        }
    )


@pytest.fixture
def experts():
    return {
        "FULL": table([-2, -3, 4, 5, 1.2, -2, 7, 12]),
        "A5": table([-1, 3, -3, 6, 2, 1.8, 4.5, 8.2]),
    }


@pytest.fixture
def scorer_spy(monkeypatch):
    seen = []

    def score(frame, prediction):
        target = frame.target_ttc.to_numpy()
        seen.append(prediction.copy())
        return {
            "mae": float(np.abs(target - prediction).mean()),
            "sign_errors": int((np.signbit(target) != np.signbit(prediction)).sum()),
            "MiD": float(diagnostics._mid(target, prediction).mean()),
            "negative_mae": float(np.abs(target[target < 0] - prediction[target < 0]).mean()),
        }

    monkeypatch.setattr(diagnostics, "score", score)
    monkeypatch.setattr(diagnostics, "gate", lambda reference, candidate: {"evaluated": True})
    return seen


def test_cross_sign_cells_are_exhaustive_disjoint_and_paired(experts):
    result = diagnostics.sign_cross_cells(experts["FULL"], experts["A5"])
    assert result["negative_count"] == 4
    assert len(result["cells"]) == 4
    assert sum(cell["count"] for cell in result["cells"].values()) == 4
    assert all(cell["count"] == 1 for cell in result["cells"].values())
    cell = result["cells"]["FULL_correct_False_comparator_correct_True"]
    # The SAME target=-4 query: FULL=+4, comparator=-3.
    assert cell["FULL"]["absolute_error_sum_s"] == 8
    assert cell["comparator"]["absolute_error_sum_s"] == 1
    assert cell["by_group"]["g1"]["FULL"]["count"] == 1
    assert cell["by_group"]["g2"]["FULL"]["count"] == 0


def test_oracle_all_metrics_use_one_selected_vector_and_ties_use_baseline(experts, scorer_spy):
    full, baseline = experts["FULL"], experts["A5"]
    full.loc[0, ["prediction_ttc", "raw_phase"]] = baseline.loc[0, ["prediction_ttc", "raw_phase"]]
    target = full.target_ttc.to_numpy()
    first = full.prediction_ttc.to_numpy()
    second = baseline.prediction_ttc.to_numpy()
    selected = diagnostics._mid(target, first) < diagnostics._mid(target, second)
    expected = np.where(selected, first, second)
    result = diagnostics.compute(experts)
    oracle = result["oracles"]["FULL_vs_A5_minimum_MiD"]
    assert selected[0] == np.False_
    assert oracle["selected_counts"] == {"FULL": int(selected.sum()), "A5": int((~selected).sum())}
    assert np.array_equal(scorer_spy[2], expected)
    assert oracle["metrics_from_one_vector"] == {
        "mae": float(np.abs(target - expected).mean()),
        "sign_errors": int((np.signbit(target) != np.signbit(expected)).sum()),
        "MiD": float(diagnostics._mid(target, expected).mean()),
        "negative_mae": float(np.abs(target[target < 0] - expected[target < 0]).mean()),
    }
    assert (
        oracle["prediction_vector_sha256_float64_le"]
        == hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest()
    )
    assert not oracle["candidate"] and not oracle["system_promotion"]
    assert len(scorer_spy) == 4  # Two fixed experts, one selection oracle, one sign oracle.


def test_sign_only_oracle_preserves_magnitude_and_is_separate(experts, scorer_spy):
    result = diagnostics.compute(experts)
    full = experts["FULL"].prediction_ttc.to_numpy()
    corrected = scorer_spy[-1]
    assert np.array_equal(np.abs(corrected), np.abs(full))
    assert np.array_equal(np.signbit(corrected), np.signbit(experts["FULL"].target_ttc))
    record = result["oracles"]["FULL_sign_only"]
    assert record["changed_sign_count"] == 3
    assert record["magnitude_preserved_exactly"]
    assert record["distinct_from_selection_oracle"]
    assert not np.array_equal(scorer_spy[-1], scorer_spy[-2])


def test_h_never_creates_any_oracles(experts, scorer_spy):
    for frame in experts.values():
        frame["role"] = "H"
    result = diagnostics.compute(experts)
    assert result["oracles"] == {}
    assert len(scorer_spy) == 2
    assert result["negative_sign_cross_vs_FULL"]["A5"]["negative_count"] == 4


@pytest.mark.parametrize("corruption", ["order", "target", "duplicate", "role", "nonfinite"])
def test_population_mismatches_are_rejected_without_dropping_rows(experts, corruption):
    if corruption == "order":
        experts["A5"] = experts["A5"].iloc[::-1]
    elif corruption == "target":
        experts["A5"].loc[0, "target_ttc"] = -2.1
    elif corruption == "duplicate":
        experts["A5"].loc[0, "sample_token"] = "query_1"
    elif corruption == "role":
        experts["FULL"]["role"] = "V"
    else:
        experts["A5"].loc[0, "prediction_ttc"] = np.nan
    with pytest.raises(ValueError):
        diagnostics.compute(experts)


def test_native_phase_saturation_includes_zero_and_does_not_use_decimal_ttc_equality():
    frame = table([-2, -3, -4, -5, 1, 2, 4, 8])
    # Use an unsaturated emitted value deliberately: the diagnostic must inspect raw phase.
    frame["raw_phase"] = [0.0, 0.0001, -0.0001, 0.01, -0.01, 0.1, -0.1, 0.2]
    saturated, proof = diagnostics._saturation(frame, "A5")
    assert saturated.tolist() == [True, True, True, False, False, False, False, False]
    assert proof["phase_reconstructed_from_prediction_caution"]
    assert proof["reconversion_max_abs_difference_s"] > 1


def test_native_scorer_preserves_registered_group_macro_weights(experts):
    result = diagnostics.compute(experts)
    oracle = result["oracles"]["FULL_vs_A5_minimum_MiD"]["metrics_from_one_vector"]
    target = experts["FULL"].target_ttc.to_numpy()
    first = experts["FULL"].prediction_ttc.to_numpy()
    second = experts["A5"].prediction_ttc.to_numpy()
    selected = diagnostics._mid(target, first) < diagnostics._mid(target, second)
    prediction = np.where(selected, first, second)
    losses = diagnostics._mid(target, prediction)
    groups = experts["FULL"].group_id.to_numpy()
    masks = [(target > 0) & (target <= 3), (target > 3) & (target <= 6), target > 6, target < 0]
    expected = 0.0
    for weight, mask in zip([0.5, 0.3, 0.1, 0.1], masks, strict=True):
        expected += weight * np.mean(
            [losses[mask & (groups == group)].mean() for group in np.unique(groups[mask])]
        )
    assert oracle["overall_MiD"] == pytest.approx(expected)
    assert oracle["diagnostics"]["mae_s"] == pytest.approx(np.abs(target - prediction).mean())
    assert oracle["diagnostics"]["sign_error_rate"] == pytest.approx(
        (np.signbit(target) != np.signbit(prediction)).mean()
    )
