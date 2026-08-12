from __future__ import annotations

import pandas as pd
import pytest

from e_jepa_ttc.data.scientific_recovery_v5 import (
    build_train_only_grouped_dev,
    validate_cache_identities,
)


def _metadata() -> pd.DataFrame:
    rows = []
    for sequence_index in range(9):
        count = 4 + (sequence_index % 3)
        for sample_index in range(count):
            rows.append(
                {
                    "sequence_id": f"sequence-{sequence_index}",
                    "sample_token": f"token-{sequence_index}-{sample_index}",
                    "track_id": f"track-{sequence_index}-{sample_index // 2}",
                    "ttc_s": float(sequence_index + sample_index),
                }
            )
    return pd.DataFrame(rows)


def test_grouped_dev_is_sequence_disjoint_exhaustive_and_balanced() -> None:
    protocol = build_train_only_grouped_dev(_metadata(), folds=3, seed=20260813)
    universe = set(protocol["sequence_ids"])
    dev_counts = {sequence: 0 for sequence in universe}

    for fold in protocol["folds"]:
        train = set(fold["train_sequence_ids"])
        dev = set(fold["dev_sequence_ids"])
        assert not train & dev
        assert train | dev == universe
        assert len(train) == 6
        assert len(dev) == 3
        for sequence in dev:
            dev_counts[sequence] += 1
    assert set(dev_counts.values()) == {1}


def test_grouped_dev_is_deterministic_and_target_blind() -> None:
    metadata = _metadata()
    first = build_train_only_grouped_dev(metadata, folds=3, seed=20260813)
    permuted = metadata.copy()
    permuted["ttc_s"] = list(reversed(permuted["ttc_s"].tolist()))
    second = build_train_only_grouped_dev(permuted, folds=3, seed=20260813)

    assert first == second


def test_grouped_dev_rejects_duplicate_sample_tokens() -> None:
    metadata = _metadata()
    metadata.loc[1, "sample_token"] = metadata.loc[0, "sample_token"]

    with pytest.raises(ValueError, match="duplicate sample_token"):
        build_train_only_grouped_dev(metadata, folds=3, seed=20260813)


def test_grouped_dev_requires_expected_nine_sequence_universe() -> None:
    metadata = _metadata()
    metadata = metadata[metadata["sequence_id"] != "sequence-8"]

    with pytest.raises(ValueError, match="Expected 9 train sequences"):
        build_train_only_grouped_dev(metadata, folds=3, seed=20260813)


def test_cache_identity_validation_is_exact_and_order_independent() -> None:
    metadata = _metadata()
    identities = metadata[["sequence_id", "sample_token", "track_id"]].sample(
        frac=1.0, random_state=11
    )

    result = validate_cache_identities(metadata, identities)
    assert result["exact_sample_tokens"] is True
    assert result["exact_sequence_ids"] is True
    assert result["exact_track_ids"] is True


def test_cache_identity_validation_rejects_track_mismatch() -> None:
    metadata = _metadata()
    identities = metadata[["sequence_id", "sample_token", "track_id"]].copy()
    identities.loc[0, "track_id"] = "wrong-track"

    with pytest.raises(ValueError, match="track identity"):
        validate_cache_identities(metadata, identities)
