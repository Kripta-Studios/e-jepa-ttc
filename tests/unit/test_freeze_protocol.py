from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_protocol_help_is_read_only() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = repo_root / "artifacts/audit/recovery_v3/frozen_protocol.json"
    before = _sha256(output) if output.exists() else None
    result = subprocess.run(
        [sys.executable, "scripts/freeze_protocol.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    after = _sha256(output) if output.exists() else None
    assert result.returncode == 0
    assert "--protocol" in result.stdout
    assert before == after
