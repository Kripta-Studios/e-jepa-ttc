import json
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash

def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent

def load_frozen_protocol() -> Dict[str, Any]:
    """
    Loads the frozen protocol from artifacts/audit/recovery_v3/frozen_protocol.json.
    Raises ValueError if it doesn't exist or if its signature is invalid.
    """
    frozen_path = get_repo_root() / "artifacts" / "audit" / "recovery_v3" / "frozen_protocol.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"Frozen protocol not found at {frozen_path}. Run scripts/freeze_protocol.py first.")
    
    with open(frozen_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not verify_artifact_hash(data):
        raise ValueError("Frozen protocol signature mismatch (artifact_sha256).")
        
    return data

def get_current_protocol_identity() -> Tuple[str, str]:
    """
    Returns (protocol_version, protocol_sha256) from the frozen protocol.
    Used by all producers to ensure they reference a verified, static protocol.
    """
    protocol = load_frozen_protocol()
    return protocol["protocol_version"], protocol["protocol_sha256"]
