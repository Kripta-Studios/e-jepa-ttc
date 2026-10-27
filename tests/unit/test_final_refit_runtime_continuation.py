"""A continuation plan must never silently restart its migrated fit."""

import json
from pathlib import Path

import pytest

from operational.final_refit.producers import _requires_runtime_migration
from operational.rgb_port.recipe import file_sha256


def _plan(tmp_path: Path) -> Path:
    admission = tmp_path / "RUNTIME_MIGRATION_ADMISSION.json"
    admission.write_text(
        json.dumps(
            {
                "schema": "final_refit_runtime_migration_admission_v1",
                "status": "ADMITTED_TECHNICAL_RUNTIME_MIGRATION",
                "fit_id": "R_A5",
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "PRODUCER_PLAN.json"
    plan.write_text(
        json.dumps({"input_sha256": {"runtime_migration_admission": file_sha256(admission)}}),
        encoding="utf-8",
    )
    return plan


def test_continuation_refuses_fresh_start_but_preserves_other_fresh_stage(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(ValueError, match="requires its exact migration receipt"):
        _requires_runtime_migration(tmp_path, plan, "R_A5")
    assert not _requires_runtime_migration(tmp_path, plan, "R_C2F")
    receipt = tmp_path / "fits" / "R_A5" / "RUNTIME_MIGRATION.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    # This helper only routes; resolve_runtime_checkpoint_identity validates the receipt.
    assert _requires_runtime_migration(tmp_path, plan, "R_A5")


def test_changed_admission_cannot_authorize_continuation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    (tmp_path / "RUNTIME_MIGRATION_ADMISSION.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        _requires_runtime_migration(tmp_path, plan, "R_A5")


def test_undeclared_migration_is_rejected(tmp_path: Path) -> None:
    plan = tmp_path / "PRODUCER_PLAN.json"
    plan.write_text('{"input_sha256": {}}', encoding="utf-8")
    receipt = tmp_path / "fits" / "R_A5" / "RUNTIME_MIGRATION.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not declared"):
        _requires_runtime_migration(tmp_path, plan, "R_A5")
