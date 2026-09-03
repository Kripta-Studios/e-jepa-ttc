from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from e_jepa_ttc.evaluation.collision_clock_runner import (
    FrozenCollisionClockCheckpoint,
    dry_run_dag,
    load_runner_contracts,
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
