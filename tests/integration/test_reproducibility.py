import json
from pathlib import Path

import jsonschema


def test_completion_manifest_v3_schema():
    repo_root = Path(__file__).resolve().parent.parent.parent
    schema_path = repo_root / "schemas" / "completion_manifest_v3.schema.json"
    
    assert schema_path.exists(), "completion_manifest_v3 schema is missing"
    
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
        
    # Verify it can reject an empty/mocked manifest
    mocked_manifest = {}
    try:
        jsonschema.validate(instance=mocked_manifest, schema=schema)
        assert False, "Schema allowed an empty dict"
    except jsonschema.ValidationError:
        pass
        
    # Verify it accepts a valid manifest
    valid_manifest = {
        "artifact_type": "completion_manifest_v3",
        "smoke_completed": True,
        "full_completed": False,
        "failures": []
    }
    
    jsonschema.validate(instance=valid_manifest, schema=schema)
