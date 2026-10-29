"""Unit tests for A4 train-only DINO lambda selection protocol."""

from __future__ import annotations

from typing import Any

import pytest
from torch.utils.data import Dataset

from scripts.select_a4_dinov3_relational_weight_cv import (
    IndexedDataset,
    _aggregate_candidate,
    _select_best_candidate,
    _validate_fold_protocol,
)


class _GroupedDataset(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"index": index}

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        return ((0, 1, 2, 3), (4, 5, 6, 7))


def test_indexed_dataset_remaps_shard_groups() -> None:
    subset = IndexedDataset(_GroupedDataset(), [1, 3, 4, 7])
    assert len(subset) == 4
    assert subset[0]["index"] == 1
    assert subset.shard_index_groups() == ((0, 1), (2, 3))


def test_fold_protocol_holds_each_sequence_once() -> None:
    sequences = [f"s{i}" for i in range(9)]
    folds = [
        {"name": "f1", "seed": 7, "heldout_sequences": ["s0", "s3", "s6"]},
        {"name": "f2", "seed": 13, "heldout_sequences": ["s1", "s4", "s7"]},
        {"name": "f3", "seed": 23, "heldout_sequences": ["s2", "s5", "s8"]},
    ]
    _validate_fold_protocol(sequences, folds)
    bad = [dict(fold) for fold in folds]
    bad[2] = {"name": "f3", "seed": 23, "heldout_sequences": ["s2", "s5", "s7"]}
    with pytest.raises(ValueError, match="exactly once|more than once"):
        _validate_fold_protocol(sequences, bad)


def test_candidate_aggregation_and_selection_tie_break() -> None:
    sequences = [f"s{i}" for i in range(9)]
    fold_results = []
    for fold_index in range(3):
        heldout = sequences[fold_index * 3 : (fold_index + 1) * 3]
        fold_results.append(
            {
                "num_samples": 100 + fold_index,
                "failure_rate_pct": 10.0 + fold_index,
                "per_sequence": {
                    sequence: {"paper_MiD_overall": 100.0 + int(sequence[1:])}
                    for sequence in heldout
                },
            }
        )
    aggregate = _aggregate_candidate(8.0, fold_results, sequences)
    assert aggregate["nine_sequence_macro_MiD"] == pytest.approx(104.0)
    expected_failure = (10.0 * 100 + 11.0 * 101 + 12.0 * 102) / 303
    assert aggregate["sample_weighted_failure_rate_pct"] == pytest.approx(expected_failure)

    best = _select_best_candidate(
        [
            {
                "lambda": 8.0,
                "nine_sequence_macro_MiD": 100.0,
                "sample_weighted_failure_rate_pct": 5.0,
            },
            {
                "lambda": 6.0,
                "nine_sequence_macro_MiD": 100.0,
                "sample_weighted_failure_rate_pct": 5.0,
            },
            {
                "lambda": 4.0,
                "nine_sequence_macro_MiD": 101.0,
                "sample_weighted_failure_rate_pct": 1.0,
            },
        ]
    )
    assert float(best["lambda"]) == 6.0
