from copy import deepcopy

import pytest

from operational.evttc_transfer_audit.protocol import validate_protocol


def record(name: str, group: str, role: str = "fit") -> dict:
    return {
        "sequence_id": name,
        "recording_group": name,
        "acquisition_group": group,
        "role": role,
        "review_status": "approved",
        "relationship_evidence": ["metadata.json"],
        "historical_exposure": ["scores_inspected"],
    }


def test_same_acquisition_cannot_cross_roles_or_escape_exclusion() -> None:
    records = [record("a", "session", "excluded"), record("b", "session")]
    result = validate_protocol({}, {"records": records}, {"excluded_recordings": ["a"]}, {})
    assert result["status"] == "rejected"
    assert result["effective_exclusions"] == ["a", "b"]


def test_alias_chain_propagates_recording_exclusion() -> None:
    records = [record("a", "s1", "excluded"), record("b", "s2"), record("c", "s3")]
    records[1]["aliases"] = ["a"]
    records[2]["aliases"] = ["b"]
    result = validate_protocol({}, {"records": records}, {"excluded_recordings": ["a"]}, {})
    assert result["effective_exclusions"] == ["a", "b", "c"]
    assert result["status"] == "rejected"


def test_unknown_relations_never_certify_independence() -> None:
    records = [record("a", "unknown", "development_exposed")]
    result = validate_protocol({}, {"records": records}, {"excluded_recordings": []}, {})
    assert result["status"] == "unresolved"
    assert result["training_allowed"] is False
    assert result["independent_groups"] is None


def test_historical_scores_prevent_clean_holdout_claim() -> None:
    records = [record("a", "session", "holdout_clean")]
    result = validate_protocol({}, {"records": records}, {"excluded_recordings": []}, {})
    assert result["status"] == "rejected"


def test_duplicate_and_missing_record_ids_are_rejected() -> None:
    a = record("a", "s")
    with pytest.raises(ValueError, match="unique"):
        validate_protocol({}, {"records": [a, deepcopy(a)]}, {}, {})


def test_policy_can_forbid_training_even_with_verified_inputs() -> None:
    result = validate_protocol(
        {"training_allowed": False},
        {"records": [record("a", "s")]},
        {"status": "verified", "excluded_recordings": []},
        {"status": "verified"},
    )
    assert result["training_allowed"] is False


def test_missing_identity_receipt_cannot_open_training() -> None:
    result = validate_protocol(
        {"training_allowed": True}, {"records": [record("a", "s")]}, {}, {"status": "verified"}
    )
    assert result["status"] == "unresolved"
    assert result["training_allowed"] is False


@pytest.mark.parametrize("history", [[], ["design_use_unknown"], ["unknown"]])
def test_status_only_receipts_and_unknown_histories_cannot_open_training(history) -> None:
    row = record("a", "session")
    row["historical_exposure"] = history
    result = validate_protocol(
        {"training_allowed": True},
        {"records": [row]},
        {"status": "verified"},
        {"status": "verified"},
    )
    assert result["status"] == "unresolved"
    assert result["training_allowed"] is False
