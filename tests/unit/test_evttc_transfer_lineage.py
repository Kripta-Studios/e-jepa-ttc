import pytest

from operational.evttc_transfer_audit.lineage import SCOPES, evaluate_ancestry


@pytest.mark.parametrize("groups", [[], ["unknown"], ["design_use_unknown"]])
def test_complete_declarations_without_verified_evidence_cannot_be_clear(groups) -> None:
    component = {scope: groups for scope in SCOPES}
    component.update(
        ancestry_complete=True,
        evidence_refs=["nonexistent-receipt"],
        parent_artifact_hashes=[],
        teacher_artifact_hash=None,
    )
    assert evaluate_ancestry(component, set())["exposure_status"] == "unresolved"


def test_evidence_must_exist_and_match_hash(tmp_path) -> None:
    from operational.evttc_transfer_audit.lineage import verify_evidence

    with pytest.raises(FileNotFoundError):
        verify_evidence(
            tmp_path,
            {"evidence_refs": ["absent.json"], "evidence_sha256": {"absent.json": "a" * 64}},
        )
    (tmp_path / "receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_evidence(
            tmp_path,
            {"evidence_refs": ["receipt.json"], "evidence_sha256": {"receipt.json": "a" * 64}},
        )


def test_frozen_weights_and_empty_lists_do_not_prove_clean_lineage() -> None:
    result = evaluate_ancestry(
        {
            "artifact_sha256": "a" * 64,
            "frozen": True,
            "training_groups": [],
            "parent_artifact_hashes": [],
        },
        set(),
    )
    assert result["exposure_status"] == "unresolved"


def test_excluded_group_in_selection_is_exposure() -> None:
    result = evaluate_ancestry({"checkpoint_selection_groups": ["competition"]}, {"competition"})
    assert result["exposure_status"] == "exposed"


def test_missing_parent_stays_unknown_even_when_child_is_documented() -> None:
    component = {
        "parent_artifact_hashes": ["b" * 64],
        "ancestry_complete": True,
        "evidence_refs": ["receipt"],
        "training_groups": [],
        "checkpoint_selection_groups": [],
        "architecture_selection_groups": [],
        "normalization_fit_groups": [],
        "teacher_artifact_hash": None,
    }
    assert evaluate_ancestry(component, set())["exposure_status"] == "unresolved"
