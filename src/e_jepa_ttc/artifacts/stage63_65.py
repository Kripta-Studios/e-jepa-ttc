"""Shared signed metadata for new Stage 63--65 scientific artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import sign_artifact


def sign_stage63_65_artifact(
    value: dict[str, Any], *, evidence_type: str, repository_root: Path | None = None
) -> dict[str, Any]:
    """Add the common scientific envelope and its canonical artifact signature."""

    root = repository_root or Path(__file__).resolve().parents[3]
    protocol_path = root / "configs" / "protocol" / "scientific_recovery_v9_stage63_65.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    value.update(
        {
            "schema_version": "1.0",
            "evidence_type": evidence_type,
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "protocol_version": protocol["protocol_id"],
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    return sign_artifact(value)


__all__ = ["sign_stage63_65_artifact"]
