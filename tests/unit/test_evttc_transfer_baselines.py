import pytest

from operational.evttc_transfer_audit.baselines import saved_predictions


def test_saved_baseline_must_match_query_clock_not_just_row_number() -> None:
    targets = [{"query_id": "a:0", "query_us": 10, "sequence_id": "a"}]
    ledger = [
        {
            "identity": {"sequence_id": "a", "row_index": 0, "anchor_us": 11},
            "prediction_ttc_seconds": 1.0,
            "variant": "full",
        }
    ]
    with pytest.raises(ValueError, match="clock"):
        saved_predictions(ledger, targets, "Garl_full")


def test_saved_baseline_keeps_absent_outputs() -> None:
    targets = [{"query_id": "a:0", "query_us": 10, "sequence_id": "a"}]
    ledger = [
        {
            "identity": {"sequence_id": "a", "row_index": 0, "anchor_us": 10},
            "prediction_ttc_seconds": None,
            "variant": "full",
        }
    ]
    assert saved_predictions(ledger, targets, "Garl_full") == [
        {"query_id": "a:0", "prediction": None}
    ]
