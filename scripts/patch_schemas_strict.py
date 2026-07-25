import json
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

def main():
    schemas_dir = Path("schemas")
    schema_files = list(schemas_dir.glob("*.schema.json"))

    for schema_file in schema_files:
        if schema_file.name == "recovery_v3_protocol.schema.json":
            # Protocol schema doesn't need to be an artifact
            continue

        with open(schema_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "properties" not in data:
            data["properties"] = {}
        if "required" not in data:
            data["required"] = []

        # Make root additionalProperties strict
        data["additionalProperties"] = False

        # Add all required fields to root
        for field in REQUIRED_ROOT_FIELDS:
            if field not in data["properties"]:
                data["properties"][field] = {"type": "string"}
            if field not in data["required"]:
                data["required"].append(field)

        # Specifically allow some nested structures to be open based on artifact logic
        artifact_type = data["properties"].get("artifact_type", {}).get("enum", [None])[0]

        # Let regression be open just in case, but keep root closed
        if artifact_type == "calibration_metrics_v3":
            if "regression" in data["properties"]:
                data["properties"]["regression"]["additionalProperties"] = True
            if "calibration" in data["properties"]:
                data["properties"]["calibration"]["additionalProperties"] = True

        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {schema_file.name}")

if __name__ == "__main__":
    main()
