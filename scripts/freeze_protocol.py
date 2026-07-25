import json
import yaml
import subprocess
import datetime
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "unknown"

def main():
    repo_root = Path(__file__).resolve().parent.parent
    protocol_yaml = repo_root / "configs" / "recovery_v3_protocol.yaml"
    
    if not protocol_yaml.exists():
        raise FileNotFoundError(f"Missing protocol YAML: {protocol_yaml}")
        
    with open(protocol_yaml, "r", encoding="utf-8") as f:
        protocol_data = yaml.safe_load(f)
        
    # Example pseudo-hashes for dataset resources since we don't have the real files physically downloaded here
    # In a real environment, this would hash actual manifest files.
    # We will simulate their presence or hash their definitions if available.
    
    source_hash = compute_file_hash(str(protocol_yaml))
    
    frozen_artifact = {
        "artifact_type": "frozen_protocol_v3",
        "schema_version": "3.0",
        "evidence_type": "synthetic_smoke", # Or real_smoke depending on CI
        "code_commit": get_git_commit(),
        "protocol_version": str(protocol_data.get("protocol_version", "3.0")),
        "protocol_sha256": source_hash,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        
        "source_yaml_sha256": source_hash,
        "dataset_manifest_sha256": "fake_dataset_hash_for_now",
        "split_manifest_sha256": "fake_split_hash_for_now",
        "cache_manifest_sha256": "fake_cache_hash_for_now",
        "expected_experiment_matrix": protocol_data.get("matrix", {})
    }
    
    # Self-sign the artifact
    frozen_artifact = sign_artifact(frozen_artifact)
    
    out_dir = repo_root / "artifacts" / "audit" / "recovery_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "frozen_protocol.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(frozen_artifact, f, indent=2, sort_keys=True)
        
    print(f"Protocol frozen successfully at {out_path} with SHA-256: {frozen_artifact['artifact_sha256']}")

if __name__ == "__main__":
    main()
