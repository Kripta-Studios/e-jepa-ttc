import pytest

from operational.evttc_transfer_audit.scoring import score_rows
from operational.evttc_transfer_audit.targets import align_targets

CONTRACT = {
    "unit": "seconds",
    "timestamp_unit": "microseconds",
    "reference": "query",
    "geometry": "local_csv_reference_unverified",
    "sign": "signed",
    "interpolation": "linear_ttc",
    "zero_policy": "mae_only",
    "max_gap_us": 20,
    "official_equivalence": "unverified",
}


def queries(times: list[int]) -> list[dict]:
    return [
        {"query_id": str(i), "sequence_id": "a", "group": "a", "query_us": t}
        for i, t in enumerate(times)
    ]


def test_interior_exact_endpoints_and_no_extrapolation() -> None:
    result = align_targets(
        queries([0, 5, 10, 11]), {"a": [[100, 2.0], [110, 0.0]]}, {"a": 100}, CONTRACT
    )
    assert [row["target_ttc"] for row in result] == [2.0, 1.0, 0.0, None]
    assert result[-1]["reason"] == "outside_label_support"
    assert result[2]["target_status"] == "valid"


@pytest.mark.parametrize("labels", [[[10, 1], [10, 2]], [[11, 1], [10, 2]]])
def test_duplicate_and_nonmonotone_labels_rejected(labels: list) -> None:
    with pytest.raises(ValueError, match="increasing"):
        align_targets(queries([10]), {"a": labels}, {"a": 0}, CONTRACT)


def test_nonfinite_gaps_and_missing_labels_stay_missing() -> None:
    rows = align_targets(
        queries([0, 5, 25]), {"a": [[0, float("nan")], [10, 1], [40, 2]]}, {"a": 0}, CONTRACT
    )
    assert [r["reason"] for r in rows] == ["nonfinite_label", "nonfinite_label", "label_gap"]
    assert align_targets(queries([0]), {}, {}, CONTRACT)[0]["reason"] == "missing_labels"


def test_signed_targets_and_clock_translation() -> None:
    first = align_targets(queries([5]), {"a": [[0, -2.0], [10, -1.0]]}, {"a": 0}, CONTRACT)
    shifted = align_targets(queries([105]), {"a": [[0, -2.0], [10, -1.0]]}, {"a": -100}, CONTRACT)
    assert first[0]["target_ttc"] == shifted[0]["target_ttc"] == -1.5


def test_macro_micro_missing_predictions_and_zero_target() -> None:
    rows = align_targets(queries([0, 1, 2, 3]), {"a": [[0, 1.0], [3, 1.0]]}, {"a": 0}, CONTRACT)
    rows[2]["group"] = "b"
    rows[3]["target_ttc"] = 0.0
    predictions = [{"query_id": str(i), "prediction": p} for i, p in enumerate([1, 1, 4, None])]
    result = score_rows(predictions, rows, {"scope": "diagnostic_partial"})
    assert result["counts"] == {
        "requested": 4,
        "finite_prediction": 3,
        "compatible_target": 4,
        "scored": 3,
        "relative_scored": 3,
    }
    assert result["metrics"]["micro"]["relative_absolute_error_pct"] == 100
    assert result["metrics"]["macro"]["relative_absolute_error_pct"] == 150
    assert result["missing_reasons"]["missing_prediction"] == 1
    assert result["official_equivalence"] == "unverified"


def test_partial_cannot_be_promoted_to_official_or_hide_unknown_predictions() -> None:
    rows = align_targets(queries([0]), {"a": [[0, 1]]}, {"a": 0}, CONTRACT)
    with pytest.raises(ValueError, match="official"):
        score_rows([], rows, {"scope": "equivalent_official"})
    with pytest.raises(ValueError, match="population"):
        score_rows([{"query_id": "other", "prediction": 1}], rows, {})
