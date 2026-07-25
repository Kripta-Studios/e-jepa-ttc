import json
import pytest
import jsonschema
from pathlib import Path

REQUIRED_ROOT_FIELDS = [
    "artifact_type",
    "schema_version",
    "evidence_type",
    "code_commit",
    "protocol_version",
    "protocol_sha256",
    "created_at",
    "artifact_sha256"
]

def get_schemas():
    repo_root = Path(__file__).resolve().parent.parent.parent
    schemas_dir = repo_root / "schemas"
    return [p for p in schemas_dir.glob("*.schema.json") if p.name != "recovery_v3_protocol.schema.json"]

@pytest.mark.parametrize("schema_path", get_schemas())
def test_schema_requires_scientific_metadata(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema.get("additionalProperties") is False, f"{schema_path.name} allows additionalProperties"
    
    required = set(schema.get("required", []))
    for field in REQUIRED_ROOT_FIELDS:
        assert field in required, f"{schema_path.name} is missing required field {field}"

def test_schema_mutation_fails():
    repo_root = Path(__file__).resolve().parent.parent.parent
    schema_path = repo_root / "schemas" / "completion_manifest_v3.schema.json"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        
    valid_instance = {
        "artifact_type": "completion_manifest_v3",
        "schema_version": "3.0",
        "evidence_type": "real_smoke",
        "code_commit": "abcdef1",
        "protocol_version": "3.0",
        "protocol_sha256": "hash",
        "created_at": "2026-07-25",
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
        "failures": []
    }
    
    # Should pass
    jsonschema.validate(instance=valid_instance, schema=schema)
    
    # Mutation tests
    for field in REQUIRED_ROOT_FIELDS:
        mutated = valid_instance.copy()
        del mutated[field]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=mutated, schema=schema)
