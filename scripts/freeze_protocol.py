import datetime
import json
import subprocess
from pathlib import Path

import yaml
from jsonschema import validate

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_dirty_worktree() -> bool:
    """Return whether tracked or untracked files differ from the commit."""

    status = subprocess.check_output(["git", "status", "--porcelain"], text=True, encoding="utf-8")
    return bool(status.strip())


def _hash_resources(repo_root: Path, resources: object) -> dict[str, dict[str, object]]:
    if not isinstance(resources, dict) or not resources:
        raise ValueError("Protocol must declare a non-empty 'resources' mapping.")
    hashed: dict[str, dict[str, object]] = {}
    for name, raw_path in sorted(resources.items()):
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Protocol resource {name!r} must be a non-empty path.")
        path = (repo_root / raw_path).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Protocol resource escapes repository: {raw_path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Protocol resource does not exist: {raw_path}")
        hashed[name] = {
            "path": path.relative_to(repo_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": compute_file_hash(str(path)),
        }
    return hashed


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    protocol_yaml = repo_root / "configs" / "recovery_v3_protocol.yaml"

    if not protocol_yaml.exists():
        raise FileNotFoundError(f"Missing protocol YAML: {protocol_yaml}")

    with protocol_yaml.open("r", encoding="utf-8") as f:
        protocol_data = yaml.safe_load(f)

    schema_path = repo_root / "schemas" / "recovery_v3_protocol.schema.json"
    with schema_path.open("r", encoding="utf-8") as f:
        validate(instance=protocol_data, schema=json.load(f))

    source_hash = compute_file_hash(str(protocol_yaml))
    resource_hashes = _hash_resources(repo_root, protocol_data.get("resources"))
    declared_manifest_hash = protocol_data["requirements"]["dataset"]["manifest_hash"]
    actual_manifest_hash = resource_hashes["evttc_dataset_manifest"]["sha256"]
    if declared_manifest_hash != actual_manifest_hash:
        raise ValueError(
            "requirements.dataset.manifest_hash does not match the declared EvTTC manifest."
        )

    frozen_artifact = {
        "artifact_type": "frozen_protocol_v3",
        "schema_version": "3.0",
        "evidence_type": "protocol_definition",
        "code_commit": get_git_commit(),
        "dirty_worktree": get_dirty_worktree(),
        "protocol_version": str(protocol_data.get("protocol_version", "3.0")),
        "claim_level": protocol_data["claim_level"],
        "test_status": protocol_data["test_status"],
        "cache_format_version": protocol_data["cache_format_version"],
        "protocol_sha256": source_hash,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_yaml_path": protocol_yaml.relative_to(repo_root).as_posix(),
        "source_yaml_sha256": source_hash,
        "resources": resource_hashes,
        "expected_experiment_matrix": protocol_data.get("matrix", {}),
        "expected_eap_matrix": protocol_data.get("eap_matrix", {}),
    }

    # Self-sign the artifact
    frozen_artifact = sign_artifact(frozen_artifact)

    out_dir = repo_root / "artifacts" / "audit" / "recovery_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "frozen_protocol.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(frozen_artifact, f, indent=2, sort_keys=True)

    print(
        f"Protocol frozen successfully at {out_path} with SHA-256: "
        f"{frozen_artifact['artifact_sha256']}"
    )


if __name__ == "__main__":
    main()
