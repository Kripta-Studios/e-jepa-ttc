from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from operational.final_refit.fold_training import (
    _validate_technical_receipt,
    build_fold_training_unit,
    load_fold_training_unit,
    write_fold_training_unit,
)
from operational.final_refit.recipe import sha256


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fold_contract(root: Path, *, overlap: bool = False) -> Path:
    counts = [88_744 // 40 + (index < 88_744 % 40) for index in range(40)]
    roles = {
        "producer": list(range(24)),
        "refiner": list(range(24, 32)),
        "generation": list(range(32, 40)),
    }
    if overlap:
        roles["refiner"][0] = roles["producer"][0]
    role_refs: dict[str, object] = {}
    token = 0
    for role, group_indices in roles.items():
        rows: list[dict[str, object]] = []
        for group_index in group_indices:
            for _ in range(counts[group_index]):
                rows.append(
                    {
                        "rotation": 0,
                        "rotation_role": role,
                        "source_ordinal": token,
                        "sample_token": f"{role}-{token}",
                        "group_id": f"group-{group_index:02d}",
                    }
                )
                token += 1
        path = root / f"rotation_0_{role}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        role_refs[role] = {
            "path": path.name,
            "sha256": sha256(path),
            "queries": len(rows),
            "groups": len(set(item["group_id"] for item in rows)),
        }
    manifest = {
        "schema": "final_refit_rotation_roles_v1",
        "rotation": 0,
        "status": "VERIFIED_REFERENCE_MANIFESTS",
        "generation_ancestry_excludes_group": True,
        "roles": role_refs,
    }
    manifest_path = root / "rotation_0.json"
    _json(manifest_path, manifest)
    rotations = [{"path": manifest_path.name, "sha256": sha256(manifest_path)} for _ in range(5)]
    contract = {
        "schema": "final_refit_oof_fold_contract_v1",
        "status": "PASS",
        "training_started": False,
        "optimizer_updates": 0,
        "generation_coverage": "EACH_TRAIN40_TOKEN_EXACTLY_ONCE",
        "population": {"queries": 88_744, "groups": 40},
        "rotations": rotations,
    }
    contract_path = root / "FOLD_CONTRACT.json"
    _json(contract_path, contract)
    return contract_path


def test_fold_unit_has_only_rotation_pair_and_context_fits(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    unit = build_fold_training_unit(
        contract,
        rotation=0,
        executed_sources={str(source.resolve()): source},
    )
    assert [(stage.stage_id, stage.role) for stage in unit.stages] == [
        ("PAIR_E", "producer_pair"),
        ("PAIR_R", "producer_pair"),
        ("E_CTX", "refiner"),
        ("R_CTX", "refiner"),
    ]
    assert all(stage.stage_id != "PHASE_ONLY" for stage in unit.stages)
    assert unit.status == "PREPARED_NOT_QUEUED"
    path = tmp_path / "FOLD_TRAINING_UNIT.json"
    write_fold_training_unit(path, unit)
    assert load_fold_training_unit(path) == unit


def test_fold_unit_rejects_overlapping_ancestor_groups(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path, overlap=True)
    source = tmp_path / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap|24/8/8"):
        build_fold_training_unit(
            contract,
            rotation=0,
            executed_sources={str(source.resolve()): source},
        )


def test_fold_unit_rejects_executed_source_drift(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    unit = build_fold_training_unit(
        contract,
        rotation=0,
        executed_sources={str(source.resolve()): source},
    )
    path = tmp_path / "FOLD_TRAINING_UNIT.json"
    write_fold_training_unit(path, unit)
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executed source changed"):
        load_fold_training_unit(path)


def test_technical_endpoint_receipt_is_rejected_not_admitted() -> None:
    with pytest.raises(RuntimeError, match="scientific endpoint"):
        _validate_technical_receipt(
            {
                "status": "COMPLETE",
                "scientific_endpoint": True,
                "completed_updates": 10,
            }
        )
    _validate_technical_receipt(
        {
            "status": "IN_PROGRESS",
            "scientific_endpoint": False,
            "completed_updates": 10,
        }
    )
