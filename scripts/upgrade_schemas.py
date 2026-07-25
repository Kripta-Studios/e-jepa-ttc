import json
from pathlib import Path

def get_base_properties():
    return {
        "schema_version": {"type": "string"},
        "evidence_type": {"type": "string"},
        "code_commit": {"type": "string"},
        "protocol_version": {"type": "string"},
        "protocol_sha256": {"type": "string"},
        "created_at": {"type": "string"},
        "artifact_sha256": {"type": "string"}
    }

def get_run_properties():
    return {
        "run_id": {"type": "string"},
        "model_name": {"type": "string"},
        "model_config": {"type": "object"},
        "model_config_sha256": {"type": "string"},
        "cache_sha256": {"type": "string"},
        "split_manifest_sha256": {"type": "string"},
        "seed": {"type": "integer"},
        "navigation_mode": {"type": "string"},
        "label_fraction": {"type": "number"},
        "train_sample_count": {"type": "integer"},
        "validation_sample_count": {"type": "integer"},
        "selection_split": {"type": "string"},
        "selection_metric": {"type": "string"},
        "checkpoint_sha256": {"type": "string"},
        "final_test_opened": {"type": "boolean"}
    }

schemas_dir = Path("schemas")

run_schemas = [
    "jepa_pretrain_run_v3.schema.json",
    "object_jepa_pretrain_run_v3.schema.json",
    "supervised_run_v3.schema.json",
    "training_run_v3.schema.json"
]

all_files = list(schemas_dir.glob("*.schema.json"))

for file_path in all_files:
    if file_path.name == "recovery_v3_protocol.schema.json":
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if "properties" not in data:
        data["properties"] = {}
    if "required" not in data:
        data["required"] = []
        
    artifact_type = data["properties"].get("artifact_type", {}).get("enum", [None])[0]
    
    # Wait, some specific properties are needed for onnx_candidate, stage_record, completion_manifest
    if artifact_type == "stage_record_v3":
        data["properties"].update({
            "stage": {"type": "string"},
            "status": {"type": "string", "enum": ["started", "passed", "failed"]},
            "started_at": {"type": "string"},
            "completed_at": {"type": "string"},
            "duration_s": {"type": "number"},
            "exit_code": {"type": "integer"},
            "command": {"type": "array", "items": {"type": "string"}},
            "inputs": {"type": "array"},
            "outputs": {"type": "array"},
            "failure": {"type": ["string", "null"]}
        })
        for req in ["stage", "status", "started_at", "completed_at", "duration_s", "exit_code", "command", "inputs", "outputs"]:
            if req not in data["required"]:
                data["required"].append(req)
                
    elif artifact_type == "onnx_candidate_v3":
        data["properties"].update({
            "checkpoint_path": {"type": "string"},
            "checkpoint_sha256": {"type": "string"},
            "cache_path": {"type": "string"},
            "cache_sha256": {"type": "string"},
            "cache_sidecar_sha256": {"type": "string"},
            "model_config": {"type": "object"},
            "model_config_sha256": {"type": "string"},
            "protocol_version": {"type": "string"},
            "protocol_sha256": {"type": "string"},
            "split_manifest_sha256": {"type": "string"},
            "normalization_sha256": {"type": "string"},
            "navigation_mode": {"type": "string"},
            "selection_split": {"type": "string"},
            "selection_metric": {"type": "string"},
            "selection_metric_value": {"type": "number"},
            "code_commit": {"type": "string"}
        })
        for req in ["checkpoint_path", "checkpoint_sha256", "cache_path", "cache_sha256", "model_config_sha256", "protocol_sha256", "selection_split", "code_commit"]:
            if req not in data["required"]:
                data["required"].append(req)
                
    elif artifact_type == "calibration_metrics_v3":
        data["properties"].update({
            "evaluation_split": {"type": "string"},
            "sample_count": {"type": "integer"},
            "sample_id_hash": {"type": "string"},
            "regression": {"type": "object", "properties": {"mae_s": {"type": "number"}, "rmse_s": {"type": "number"}, "median_abs_error_s": {"type": "number"}}, "required": ["mae_s", "rmse_s", "median_abs_error_s"]},
            "risk_support": {"type": "array"},
            "calibration": {"type": "object"}
        })
        for req in ["evaluation_split", "sample_count", "sample_id_hash", "regression", "risk_support", "calibration"]:
            if req not in data["required"]:
                data["required"].append(req)
                
    else:
        # Generic updates
        base_props = get_base_properties()
        for k, v in base_props.items():
            if k not in data["properties"]:
                data["properties"][k] = v
            # Not strict requiring these for everything just yet unless it's a run schema
            
        if file_path.name in run_schemas:
            run_props = get_run_properties()
            for k, v in run_props.items():
                if k not in data["properties"]:
                    data["properties"][k] = v
            for req in run_props.keys():
                if req not in data["required"]:
                    data["required"].append(req)

    data["additionalProperties"] = True  # Too strict otherwise unless specified

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
