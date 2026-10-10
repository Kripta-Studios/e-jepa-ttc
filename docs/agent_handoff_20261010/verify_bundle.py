"""Verify the handoff's committed byte identities and local Markdown links."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    """Fail explicitly on a missing file, changed bytes or broken relative link."""
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for section in ("files", "public_inputs"):
        for relative, expected in manifest[section].items():
            path = ROOT / relative
            if not path.is_file():
                raise ValueError(f"Missing evidence: {relative}")
            data = path.read_bytes()
            if isinstance(expected, dict):
                if expected["normalization"] != "lf":
                    raise ValueError("Unsupported manifest normalization")
                data = data.replace(b"\r\n", b"\n")
                expected_hash = expected["sha256"]
            else:
                expected_hash = expected
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected_hash:
                raise ValueError(f"Changed evidence bytes: {relative}")
            checked += 1
    links = 0
    for path in sorted(HERE.rglob("*.md")):
        for raw in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if raw.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target = unquote(raw.split("#", 1)[0].strip("<>"))
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                raise ValueError(f"Broken or external local link: {path.name}: {raw}")
            links += 1
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "hashed_files": checked,
                "local_markdown_links": links,
                "gpu_seconds": 0,
                "scope": "File identity and local links; not a model rerun",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
