from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from e_jepa_ttc.evaluation.collision_clock_protocol import ROW_LEVEL_OOF_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_NAMES = {
    "reference",
    "protocol",
    "config",
    "run_manifest",
    "initialization_manifest",
    "data_cache_binding",
    "split_binding",
    "checkpoint_manifest",
    "resume_decision",
    "row_level_oof",
    "fold_summary",
    "bootstrap_artifact",
    "gate_decision",
    "aggregate",
    "dyn_w_not_executed",
}


def _schema(name: str) -> dict:
    path = ROOT / f"schemas/scientific_recovery_v9_eclock_{name}_v2.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_required_v2_schemas_are_closed_at_the_artifact_boundary() -> None:
    for name in SCHEMA_NAMES:
        schema = _schema(name)
        assert schema["additionalProperties"] is False
        assert schema["required"]
        jsonschema.Draft202012Validator.check_schema(schema)


def test_row_level_schema_matches_the_exact_export_contract() -> None:
    schema = _schema("row_level_oof")
    assert set(schema["required"]) == set(ROW_LEVEL_OOF_COLUMNS)
    assert set(schema["properties"]) == set(ROW_LEVEL_OOF_COLUMNS)


def test_closed_schema_rejects_unexpected_field() -> None:
    summary = json.loads(
        (
            ROOT
            / "configs/experiment/scientific_recovery_v9_eclock/x0_dyn_w_not_executed_summary.json"
        ).read_text(encoding="utf-8")
    )
    summary["unexpected"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_schema("dyn_w_not_executed")).validate(summary)
