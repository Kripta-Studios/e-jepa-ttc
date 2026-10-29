"""Scientific partition checks independent of any detector or GPU."""

from dataclasses import replace

import pytest

from e_jepa_ttc.detection.protocol import build_sequence_split, validate_sequence_split

SEQUENCES = tuple(f"sequence_{i:02d}" for i in range(40))
REVISION = "a814a946e60d37f7242358af5f344542177e50c5"


def candidate(sequences=SEQUENCES, **kwargs):
    return build_sequence_split(
        sequences,
        source_split="train",
        dataset_revision=REVISION,
        media_sha256="a" * 64,
        **kwargs,
    )


def test_partition_is_complete_disjoint_and_order_invariant():
    first = candidate()
    assert first == candidate(reversed(SEQUENCES))
    assert {role: len(ids) for role, ids in first.assignments.items()} == {
        "train": 24,
        "selection": 8,
        "calibration": 8,
    }
    assert first.status == "CANDIDATE_METADATA_ONLY"
    assert "not existing TRAIN40" in first.independence_scope
    assert len({s for ids in first.assignments.values() for s in ids}) == 40


def test_rejects_test_manifest_and_forbidden_sequences():
    split = candidate()
    with pytest.raises(ValueError, match="public train"):
        validate_sequence_split(replace(split, source_split="test"), eligible_sequences=SEQUENCES)
    with pytest.raises(ValueError, match="Forbidden"):
        validate_sequence_split(
            split, eligible_sequences=SEQUENCES, forbidden_sequences=SEQUENCES[:1]
        )


def test_rejects_duplicate_input_and_sequence_overlap():
    with pytest.raises(ValueError, match="duplicate"):
        candidate(SEQUENCES[:-1] + SEQUENCES[:1])
    split = candidate()
    assignments = dict(split.assignments)
    assignments["calibration"] = assignments["selection"]
    with pytest.raises(ValueError, match="overlaps"):
        validate_sequence_split(
            replace(split, assignments=assignments), eligible_sequences=SEQUENCES
        )


def test_rejects_unknown_or_omitted_sequences():
    split = candidate()
    assignments = dict(split.assignments)
    assignments["train"] = assignments["train"][:-1] + ("unlisted",)
    with pytest.raises(ValueError, match="exactly the eligible"):
        validate_sequence_split(
            replace(split, assignments=assignments), eligible_sequences=SEQUENCES
        )


def test_known_session_cannot_cross_roles():
    split = candidate()
    sessions = {sequence: sequence for sequence in SEQUENCES}
    sessions[split.assignments["train"][0]] = "same_drive"
    sessions[split.assignments["selection"][0]] = "same_drive"
    with pytest.raises(ValueError, match="crosses"):
        validate_sequence_split(split, eligible_sequences=SEQUENCES, session_groups=sessions)
    with pytest.raises(ValueError, match="exactly all"):
        validate_sequence_split(split, eligible_sequences=SEQUENCES, session_groups={})


@pytest.mark.parametrize("field,value", [("dataset_revision", "main"), ("media_sha256", "unknown")])
def test_revision_and_input_identity_are_not_optional(field, value):
    with pytest.raises(ValueError):
        validate_sequence_split(
            replace(candidate(), **{field: value}), eligible_sequences=SEQUENCES
        )


@pytest.mark.parametrize("counts", [(0, 20, 20), (24, 8, 7), (True, 19, 20)])
def test_counts_are_explicit_and_complete(counts):
    with pytest.raises(ValueError):
        candidate(train_count=counts[0], selection_count=counts[1], calibration_count=counts[2])


@pytest.mark.parametrize("seed", [-1, True, "7"])
def test_seed_is_nonnegative_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        candidate(seed=seed)
