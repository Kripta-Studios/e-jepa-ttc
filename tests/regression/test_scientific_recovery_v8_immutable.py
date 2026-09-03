from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts/packages/e-jepa-ttc-v8-essential-results-20260903.manifest.json"
EXPECTED_MANIFEST_SHA256 = "86b97a61232e6d260ee273009460981c9a1d1e75f1588a4b5c41432882ddc378"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_v8_manifest_and_every_packaged_member_are_unchanged() -> None:
    assert _sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["source_git_commit"] == "718e0bf7ca9950fbc0fc2a3537e4b0e0e25a72a2"
    assert len(payload["files"]) == 1517
    for entry in payload["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]
