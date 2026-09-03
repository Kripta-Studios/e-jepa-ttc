from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_config import (
    assert_arm_execution_authorized,
    load_x0_config,
    validate_matched_base_dyn,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/experiment/scientific_recovery_v9_eclock"
SCHEMA = ROOT / "schemas/scientific_recovery_v9_eclock_config_v1.schema.json"


def test_all_x0_configs_validate_and_protocol_signature_is_canonical() -> None:
    configs = [load_x0_config(path, schema_path=SCHEMA) for path in CONFIG_ROOT.glob("*.yaml")]
    assert {config["arm_id"] for config in configs} == {
        "X0-A5-REPLAY",
        "X0-PAIR-U",
        "X0-BASE-U",
        "X0-DYN-U",
        "X0-DYN-W",
    }
    protocol = json.loads(
        (ROOT / "configs/protocol/scientific_recovery_v9_eclock_x0.json").read_text(
            encoding="utf-8"
        )
    )
    assert verify_artifact_hash(protocol)
    protocol_schema = json.loads(
        (ROOT / "schemas/scientific_recovery_v9_eclock_protocol_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(protocol_schema).validate(protocol)


def test_base_dyn_configs_are_matched_and_dyn_w_is_nonexecutable() -> None:
    base = load_x0_config(CONFIG_ROOT / "x0_base_u.yaml", schema_path=SCHEMA)
    dynamic = load_x0_config(CONFIG_ROOT / "x0_dyn_u.yaml", schema_path=SCHEMA)
    weighted = load_x0_config(CONFIG_ROOT / "x0_dyn_w.yaml", schema_path=SCHEMA)
    validate_matched_base_dyn(base, dynamic)
    assert weighted["loss_reduction"] == "normalized_weighted_absolute_phase_error"
    with pytest.raises(PermissionError):
        assert_arm_execution_authorized(weighted)


def test_dyn_w_nonexecuted_summary_is_signed_and_schema_valid() -> None:
    summary = json.loads(
        (CONFIG_ROOT / "x0_dyn_w_not_executed_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (ROOT / "schemas/scientific_recovery_v9_eclock_artifact_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert verify_artifact_hash(summary)
    jsonschema.Draft202012Validator(schema).validate(summary)
    assert summary["loss_reduction"] == "normalized_weighted_absolute_phase_error"
