from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from torch import nn

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockOuterDevView,
    CollisionClockSampleLocator,
)
from e_jepa_ttc.evaluation.collision_clock_protocol import canonical_records_hash
from e_jepa_ttc.evaluation.collision_clock_runner import (
    FrozenCollisionClockCheckpoint,
    _freeze_official_a5,
    dry_run_dag,
    load_runner_contracts,
    replay_official_a5_outer_dev_once,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/experiment/scientific_recovery_v9_eclock"


@pytest.mark.parametrize("config_path", sorted(CONFIG_ROOT.glob("*.yaml")))
def test_every_arm_dry_run_exposes_dag_without_opening_shards_or_writing(
    config_path: Path, tmp_path: Path
) -> None:
    config, protocol, reference = load_runner_contracts(ROOT, config_path)
    output = tmp_path / config["arm_id"]
    dag = dry_run_dag(
        config=config,
        config_path=config_path,
        protocol=protocol,
        reference=reference,
        cache_root=tmp_path / "cache-is-not-opened",
        source_root=tmp_path / "reference-is-not-opened",
        output_root=output,
    )
    assert dag["opens_cache_shards"] is False
    assert dag["creates_scientific_results"] is False
    assert dag["checkpoint_policy"] == "last_update_fixed_budget"
    assert len(dag["folds"]) == 3
    assert not output.exists()


def test_evaluator_capability_cannot_be_self_declared(tmp_path: Path) -> None:
    checkpoint = tmp_path / "not-frozen.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="cannot be self-declared"):
        FrozenCollisionClockCheckpoint(
            model=nn.Linear(1, 1),
            checkpoint_path=checkpoint,
            checkpoint_file_sha256="0" * 64,
            checkpoint_manifest_sha256="1" * 64,
            external_official_a5=False,
            _capability=object(),
        )


def test_official_a5_replay_uses_only_sha_bound_oof_rows(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "sample_token": ["token-a", "token-b"],
            "sequence_id": ["sequence", "sequence"],
            "track_id": ["track-a", "track-b"],
            "target_ttc_s": [2.0, -3.0],
            "point_prediction_ttc_s": [2.25, -2.75],
            "fold": [0, 0],
            "seed": [7, 7],
        }
    )
    source_path = tmp_path / "official.csv"
    source.to_csv(source_path, index=False)
    prediction_identity = pd.DataFrame(
        {
            "sample_token": source["sample_token"].astype(str),
            "prediction_ttc_s": source["point_prediction_ttc_s"],
        }
    )
    reference = {
        "families": {
            "official_a5_oof": {
                "reference_family": "official_a5_oof",
                "row_count": 2,
                "prediction_sha256": canonical_records_hash(
                    prediction_identity, ("sample_token", "prediction_ttc_s")
                ),
                "physical_references": [
                    {
                        "semantic_identity": "official_a5_oof_csv",
                        "path": source_path.name,
                        "bytes": source_path.stat().st_size,
                        "file_sha256": compute_file_hash(str(source_path)),
                    }
                ],
            }
        }
    }
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"verified checkpoint bytes")
    frozen = _freeze_official_a5(nn.Linear(1, 1), checkpoint, compute_file_hash(str(checkpoint)))
    locators = tuple(
        CollisionClockSampleLocator(
            shard_path="unused",
            row_index=index,
            sample_token=str(source.loc[index, "sample_token"]),
            sequence_id=str(source.loc[index, "sequence_id"]),
            track_id=str(source.loc[index, "track_id"]),
            outer_fold=0,
            target_ttc_s=float(source.loc[index, "target_ttc_s"]),
            sample_weight=0.5,
        )
        for index in range(len(source))
    )
    result = replay_official_a5_outer_dev_once(
        frozen,
        CollisionClockOuterDevView(0, locators, "identity"),
        source_root=tmp_path,
        reference=reference,
        seed=7,
        config_sha256="c" * 64,
        protocol={
            "artifact_sha256": "p" * 64,
            "metric": {
                "metric_delta_t_s": 0.1,
                "deployment_ttc_clip_seconds": 60.0,
                "minimum_abs_prediction_ttc_s": 0.1,
            },
            "cache_binding": {"file_sha256": "a" * 64},
            "split_binding": {"file_sha256": "b" * 64},
        },
    )
    assert result["predicted_ttc_raw"].tolist() == [2.25, -2.75]
    assert result["arm_id"].unique().tolist() == ["X0-A5-REPLAY"]
