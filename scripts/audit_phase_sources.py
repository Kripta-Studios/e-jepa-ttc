"""Audit retained Python dependencies against the pre-replay Git baseline."""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import zipfile
from pathlib import Path

from operational.phase_adapt_cache import atomic_receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.baseline], text=True).strip()
    data = subprocess.check_output(
        [
            "git",
            "archive",
            "--format=zip",
            revision,
            "src",
            "operational",
            "scripts",
        ]
    )
    files = {}
    line_ending_only = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if not name.endswith(".py"):
                continue
            baseline_bytes = archive.read(name)
            actual = Path(name).read_bytes()
            if actual != baseline_bytes:
                if actual.replace(b"\r\n", b"\n") != baseline_bytes.replace(b"\r\n", b"\n"):
                    raise ValueError("Retained implementation differs from baseline: " + name)
                line_ending_only.append(name)
            files[name] = hashlib.sha256(actual).hexdigest()
    atomic_receipt(
        args.output,
        {
            "baseline_commit": revision,
            "files": files,
            "verification": "all retained tracked Python matches baseline allowing CRLF/LF only",
            "line_ending_only_differences": line_ending_only,
            "scope": (
                "post-run retained-checkout audit; "
                "not continuous monitoring or a pre-run full freeze"
            ),
            "additional_new_execution_sources": "already bound by replay/RUN_FREEZE.json",
        },
    )
    print(
        f"COMPLETE {len(files)} Python sources match {revision}; {len(line_ending_only)} EOL-only"
    )


if __name__ == "__main__":
    main()
