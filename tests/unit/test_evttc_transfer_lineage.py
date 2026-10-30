from operational.evttc_transfer_audit.lineage import evaluate_ancestry


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
