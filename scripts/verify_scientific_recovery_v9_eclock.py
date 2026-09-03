#!/usr/bin/env python
"""Verify E-Clock protocol and signed JSON artifacts without trusting summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_gates import validate_x0_claim_scope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not verify_artifact_hash(payload):
            raise ValueError(f"artifact signature mismatch: {path}")
        if "arm_id" in payload and payload["arm_id"] != "REFERENCE":
            validate_x0_claim_scope(payload, arm_id=str(payload["arm_id"]))
        print(f"VERIFIED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
