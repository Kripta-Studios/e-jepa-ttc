import json
from pathlib import Path

import jsonschema
import pytest


def test_completion_manifest_v3_schema():
    repo_root = Path(__file__).resolve().parent.parent.parent
    schema_path = repo_root / "schemas" / "completion_manifest_v3.schema.json"

    assert schema_path.exists(), "completion_manifest_v3 schema is missing"

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    # Verify it can reject an empty/mocked manifest
    mocked_manifest = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=mocked_manifest, schema=schema)

    # Verify it accepts a valid manifest
    valid_manifest = {
        "artifact_type": "completion_manifest_v3",
        "schema_version": "3.0",
        "evidence_type": "real_smoke",
        "code_commit": "abcdef1",
        "protocol_version": "3.0",
        "protocol_sha256": "hash",
        "created_at": "2026-07-25T00:00:00+00:00",
        "artifact_sha256": "hash",
        "status": "passed",
        "exit_code": 0,
        "all_required_artifacts_exist": True,
        "cache_v2_validation_passed": True,
        "pytorch_onnx_equivalence_passed": True,
        "final_test_opened": False,
        "required_artifact_count": 10,
        "validated_artifact_count": 10,
        "completed_stages": [],
        "failed_stages": [],
        "smoke_completed": True,
        "full_completed": False,
        "failures": [],
    }

    jsonschema.validate(instance=valid_manifest, schema=schema)
