"""Audit real H/V metadata identities without decoding pixels or evaluating labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from e_jepa_ttc.rgb_port.data import RGBPortDataset, history_observations
from operational.rgb_port.accounting import atomic_write_json, sha256_file

from .signature import observation_signature


def audit(run: Path) -> dict[str, Any]:
    """Traverse every real history and require identical inputs for repeated IDs."""
    result: dict[str, Any] = {}
    for role in ("H", "V"):
        manifest = run / f"{role}_MANIFEST.json"
        dataset = RGBPortDataset(manifest, role=role)
        seen: dict[str, str] = {}
        repeated = references = 0
        for row in dataset.rows:
            for observation in history_observations(row):
                references += 1
                key = str(observation["observation_id"])
                signature = observation_signature(observation)
                if key in seen:
                    if seen[key] != signature:
                        raise ValueError(f"{role}: repeated observation input mismatch")
                    repeated += 1
                else:
                    seen[key] = signature
        digest = hashlib.sha256(
            json.dumps(sorted(seen.items()), separators=(",", ":")).encode()
        ).hexdigest()
        result[role] = {
            "queries": len(dataset.rows),
            "unique_observations": len(seen),
            "references": references,
            "repeated_consistent": repeated,
            "identity_set_sha256": digest,
            "manifest_sha256": sha256_file(manifest),
        }
    return {**result, "status": "PASSED", "optimizer_updates": 0, "labels_used": False}


def main() -> int:
    """Write counts and identity digests, excluding samples and private paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
