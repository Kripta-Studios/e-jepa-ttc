"""Verify a complete committed source inventory without publishing a science freeze."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.freeze_integrity import FrozenFile, verify_code_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.worktree.resolve(strict=True)
    if args.output.exists() or not args.output.resolve().is_relative_to(work):
        raise ValueError("new companion evidence output required")
    started = time.monotonic()
    paths = sorted(
        path
        for folder in ("src", "scripts")
        for path in (work / folder).rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".ps1", ".psm1"}
    )
    pins = [
        FrozenFile("code", "work", path.relative_to(work).as_posix(), sha256(path))
        for path in paths
    ]
    verify_code_commit(pins, {"work": work}, args.code_commit)
    result = {
        "status": "COMPLETE_EXECUTABLE_INVENTORY_VERIFIED_NOT_SCIENTIFIC_FREEZE",
        "code_commit": args.code_commit,
        "files": [asdict(pin) for pin in pins],
        "total_bytes": sum(path.stat().st_size for path in paths),
        "observed_seconds": time.monotonic() - started,
        "optimizer_updates": 0,
        "scientific_admission": False,
        "not_covered": [
            "dependency environment",
            "data roles and producer lineage",
            "scientific QA and fit admission",
        ],
    }
    write_new_json(args.output, result)
    print(result["status"], len(pins), result["observed_seconds"])


if __name__ == "__main__":
    main()
