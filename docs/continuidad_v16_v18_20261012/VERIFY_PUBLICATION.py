"""Verify the byte-preserved published report and result payloads."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    path = (root / entry["path"]).resolve()
    assert path.is_relative_to(root), entry["path"]
    assert path.stat().st_size == entry["bytes"], entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
print(f"PASS: {len(manifest['files'])} publication payloads")
