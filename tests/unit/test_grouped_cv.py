from __future__ import annotations

from e_jepa_ttc.data.grouped_cv import create_grouped_folds, validate_grouped_folds
from e_jepa_ttc.data.types import DatasetSequence


def _sequences() -> list[DatasetSequence]:
    return [
        DatasetSequence(
            dataset="EvTTC",
            sequence_id=f"seq-{index}",
            local_path=f"data/seq-{index}",
            event_hdf5="events.h5",
            scenario_family=f"family-{index % 4}",
            speed_bucket=("low", "medium", "high")[index % 3],
            split_group=f"group-{index}",
        )
        for index in range(12)
    ]


def test_grouped_cv_is_disjoint_and_covers_every_sequence_once() -> None:
    sequences = _sequences()
    folds = create_grouped_folds(sequences, folds=5, seed=7)
    validate_grouped_folds(sequences, folds)
    validation = [sequence for fold in folds for sequence in fold["validation"]]
    assert sorted(validation) == sorted(sequence.sequence_id for sequence in sequences)


def test_grouped_cv_is_deterministic() -> None:
    assert create_grouped_folds(_sequences(), seed=7) == create_grouped_folds(
        _sequences(),
        seed=7,
    )
