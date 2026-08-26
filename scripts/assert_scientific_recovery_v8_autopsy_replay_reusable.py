#!/usr/bin/env python
"""Refuse autopsy replay reuse unless producer identity matches this HEAD.

Existence of signed manifests is not enough.  Stage 13 verifies file hashes and
protocol signatures; it does not by itself prove the replay was produced by the
current implementation on a clean worktree.  After CausalScaleTTC / provenance
repairs, missing git_commit or a mismatched commit must force stages 10-12 to
rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402
from e_jepa_ttc.scientific_provenance import (  # noqa: E402
    ScientificProvenanceError,
    assert_autopsy_replay_producer_reusable,
    require_clean_scientific_worktree,
)


class ReplayReuseError(ValueError):
    """Raised when an autopsy replay cannot be reused under the current HEAD."""


def _load_signed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReplayReuseError(f"replay manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ReplayReuseError(f"replay manifest is unsigned or corrupt: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5-manifest", type=Path, required=True)
    parser.add_argument("--c2f-manifest", type=Path, required=True)
    parser.add_argument("--garl-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        identity = require_clean_scientific_worktree()
        expected = identity["git_commit"]
        for path in (args.a5_manifest, args.c2f_manifest, args.garl_manifest):
            resolved = path if path.is_absolute() else ROOT / path
            payload = _load_signed(resolved)
            assert_autopsy_replay_producer_reusable(
                payload,
                expected_commit=expected,
                source=str(resolved),
            )
    except (OSError, json.JSONDecodeError, ReplayReuseError, ScientificProvenanceError) as error:
        print(f"autopsy replay not reusable: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "reusable", "git_commit": expected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
