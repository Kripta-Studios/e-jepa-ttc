from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheConfig
from scripts.materialize_garlttc_rotating_cache import (
    _planned_storage_bytes,
    _validate_plan,
    execute_rotating_cache,
    identity_hash,
    partition_selected_rows,
)


def _frame() -> pd.DataFrame:
    rows = []
    for role, sequence, count in (
        ("train", "train-a", 3),
        ("train", "train-b", 2),
        ("validation", "validation-a", 2),
    ):
        for index in range(count):
            rows.append(
                {
                    "sequence_id": sequence,
                    "sample_token": f"{sequence}-{index}",
                    "track_id": f"track-{index}",
                    "public_track_id": f"public-{index}",
                    "timestamp_us": index,
                    "role": role,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_partition_preserves_order_and_global_identity_coverage() -> None:
    frame = _frame()
    roles = {
        sequence: role
        for sequence, role in frame[["sequence_id", "role"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    shards = partition_selected_rows(frame, roles, shard_size=2)

    identities = [identity for shard in shards for identity in shard.identities]
    expected = [
        tuple(
            str(row[key])
            for key in (
                "sequence_id",
                "sample_token",
                "track_id",
                "public_track_id",
                "timestamp_us",
            )
        )
        for row in frame.to_dict(orient="records")
    ]
    assert identities == expected
    assert sum(len(shard.rows) for shard in shards) == len(frame)
    assert all(shard.identity_sha256 == identity_hash(shard.identities) for shard in shards)


def test_plan_validation_rejects_changed_shard_identity() -> None:
    frame = _frame()
    roles = {
        sequence: role
        for sequence, role in frame[["sequence_id", "role"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    shards = partition_selected_rows(frame, roles, shard_size=2)
    plan = {
        "artifact_type": "garlttc_rotating_cache_plan_v1",
        "status": "pass",
        "materialization_started": False,
        "garlttc_data_sha256": "data",
        "garlttc_annotations_sha256": "annotations",
        "shards": [
            {
                "role": shard.role,
                "shard_index": shard.shard_index,
                "row_count": len(shard.rows),
                "row_identity_sha256": shard.identity_sha256,
            }
            for shard in shards
        ],
    }
    plan["shards"][0]["row_identity_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_plan(plan, shards, data_sha256="data", annotations_sha256="annotations")


def test_deletion_requires_a_real_consumer() -> None:
    with pytest.raises(ValueError, match="consumer callback"):
        execute_rotating_cache(
            eap_root=Path("E:/missing-eap"),
            garlttc_root=Path("E:/missing-garl"),
            split_path=Path("E:/missing-split.json"),
            plan_path=Path("E:/missing-plan.json"),
            output_dir=Path("E:/missing-output"),
            config=GarlTTCLHRCacheConfig(),
            selection_seed=7,
            shard_size=2,
            expected_rows=1,
            workers=1,
            delete_after_consume=True,
        )


def test_unbounded_retained_execution_fails_before_source_access() -> None:
    with pytest.raises(ValueError, match="Unbounded retained-cache execution is unsafe"):
        execute_rotating_cache(
            eap_root=Path("E:/missing-eap"),
            garlttc_root=Path("E:/missing-garl"),
            split_path=Path("E:/missing-split.json"),
            plan_path=Path("E:/missing-plan.json"),
            output_dir=Path("E:/missing-output"),
            config=GarlTTCLHRCacheConfig(),
            selection_seed=7,
            shard_size=256,
            expected_rows=1,
            workers=1,
        )


def test_storage_preflight_distinguishes_retained_total_from_rotating_peak() -> None:
    plan = {
        "shards": [
            {"estimated_bytes": 100},
            {"estimated_bytes": 250},
            {"estimated_bytes": 50},
        ]
    }
    assert _planned_storage_bytes(plan, max_shards=None, retained=True) == 400
    assert _planned_storage_bytes(plan, max_shards=None, retained=False) == 250
    assert _planned_storage_bytes(plan, max_shards=2, retained=True) == 350
