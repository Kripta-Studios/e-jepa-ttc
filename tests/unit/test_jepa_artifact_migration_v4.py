from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate


def _migration_module():
    path = Path(__file__).parents[2] / "scripts" / "repair_eap_geo2_provenance.py"
    spec = importlib.util.spec_from_file_location("repair_eap_geo2_provenance", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load migration script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_consumes_real_artifact_and_preserves_source_hash(tmp_path: Path) -> None:
    source_path = tmp_path / "metrics.json"
    source = {
        "artifact_type": "eap_ssl_on_demand_pretraining_v1",
        "pretraining_regime": "eap_ssl",
        "status": "failed",
        "failure": {"type": "RuntimeError", "message": "fixture"},
    }
    source_path.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    output_path = tmp_path / "migrated.json"
    module = _migration_module()

    planned = module.migrate(source_path, output_path, dry_run=True)
    assert planned["status"] == "planned"
    assert not output_path.exists()

    report = module.migrate(source_path, output_path)
    assert report["status"] == "migrated"
    migrated = json.loads(output_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (Path(__file__).parents[2] / "schemas" / "jepa_pretrain_run_v4.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validate(migrated, schema)
    missing_flag = dict(migrated)
    missing_flag["provenance"] = dict(migrated["provenance"])
    del missing_flag["provenance"]["uses_masks_for_sampling"]
    with pytest.raises(ValidationError):
        validate(missing_flag, schema)
    leaking = dict(migrated)
    leaking["provenance"] = dict(migrated["provenance"])
    leaking["provenance"]["uses_ttc_for_sampling"] = True
    with pytest.raises(ValidationError):
        validate(leaking, schema)
    assert migrated["status"] == "failed"
    assert migrated["source_sha256"] == report["source_sha256"]
    assert (tmp_path / "artifact_migration_v4.json").is_file()
