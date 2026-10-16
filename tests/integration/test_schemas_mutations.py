import json
from pathlib import Path

import jsonschema
import pytest

REQUIRED_ROOT_FIELDS = [
    "artifact_type",
    "schema_version",
    "evidence_type",
    "code_commit",
    "protocol_version",
    "protocol_sha256",
    "created_at",
    "artifact_sha256",
]


def get_schemas(*, eclock=False):
    repo_root = Path(__file__).resolve().parent.parent.parent
    schemas_dir = repo_root / "schemas"
    return [
        p
        for p in schemas_dir.glob("*.schema.json")
        if p.name != "recovery_v3_protocol.schema.json"
        and p.name.startswith("scientific_recovery_v9_eclock_") == eclock
    ]


@pytest.mark.parametrize("schema_path", get_schemas())
def test_schema_requires_scientific_metadata(schema_path):
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    assert schema.get("additionalProperties") is False, (
        f"{schema_path.name} allows additionalProperties"
    )

    required = set(schema.get("required", []))
    for field in REQUIRED_ROOT_FIELDS:
        assert field in required, f"{schema_path.name} is missing required field {field}"


@pytest.mark.parametrize("schema_path", get_schemas(eclock=True))
def test_eclock_schema_requires_its_native_identity(schema_path):
    """V9 rows/configs are not V3 artifact envelopes; retain their versioned contracts."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    required = set(schema["required"])
    if "row_level_oof" in schema_path.name:
        assert {
            "sample_token",
            "sequence_id",
            "seed",
            "checkpoint_sha256",
            "protocol_sha256",
        } <= required
    elif "_config_" in schema_path.name:
        assert {"artifact_type", "seed", "execution_authorized", "sealed_evaluation"} <= required
    else:
        assert {"artifact_type", "artifact_sha256"} <= required
    # These two historical v1 envelopes intentionally allowed extensions. V2 closes them.
    extensible_v1 = {
        "scientific_recovery_v9_eclock_artifact_v1.schema.json",
        "scientific_recovery_v9_eclock_cross_arm_v1.schema.json",
    }
    assert schema.get("additionalProperties", True) is (schema_path.name in extensible_v1)


def test_legacy_eclock_smoke_cannot_be_relabelled_scientific():
    path = (
        Path(__file__).resolve().parents[2]
        / "schemas/scientific_recovery_v9_eclock_artifact_v1.schema.json"
    )
    schema = json.loads(path.read_text(encoding="utf-8"))
    value = {
        "artifact_type": "eclock_x0_summary_v1",
        "artifact_sha256": "a" * 64,
        "arm_id": "X0-DYN-U",
        "evidence_class": "smoke",
        "scientific_result": False,
        "upstream_roi_is_box_conditioned": True,
        "explicit_foreground_height_interface_bypassed": True,
        "loss_reduction": "mean_smooth_l1_benchmark_phase_error",
    }
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(value)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(value | {"scientific_result": True})


def test_schema_mutation_fails():
    repo_root = Path(__file__).resolve().parent.parent.parent
    schema_path = repo_root / "schemas" / "completion_manifest_v3.schema.json"

    with open(schema_path, encoding="utf-8") as f:
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
        "failures": [],
    }

    # Should pass
    jsonschema.validate(instance=valid_instance, schema=schema)

    # Mutation tests
    for field in REQUIRED_ROOT_FIELDS:
        mutated = valid_instance.copy()
        del mutated[field]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=mutated, schema=schema)
