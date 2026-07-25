import json
import hashlib
from typing import Any, Dict

def canonical_json(data: Dict[str, Any]) -> bytes:
    """Serializes a dictionary into canonical JSON for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def compute_artifact_hash(data: Dict[str, Any]) -> str:
    """
    Computes the SHA-256 hash of a scientific artifact.
    It ignores the 'artifact_sha256' field if it is present.
    """
    temp_data = dict(data)
    if "artifact_sha256" in temp_data:
        del temp_data["artifact_sha256"]
    
    canonical_bytes = canonical_json(temp_data)
    return hashlib.sha256(canonical_bytes).hexdigest()

hash_dict = compute_artifact_hash

def sign_artifact(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the hash of the artifact and assigns it to 'artifact_sha256'.
    Returns the modified dictionary.
    """
    data["artifact_sha256"] = compute_artifact_hash(data)
    return data

def verify_artifact_hash(data: Dict[str, Any]) -> bool:
    """
    Verifies if the 'artifact_sha256' matches the actual hash of the data.
    """
    expected_hash = data.get("artifact_sha256")
    if not expected_hash:
        return False
    return compute_artifact_hash(data) == expected_hash

def compute_file_hash(path: str) -> str:
    """Computes SHA-256 for a physical file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
