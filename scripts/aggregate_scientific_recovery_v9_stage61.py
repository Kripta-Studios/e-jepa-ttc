"""Create a signed inventory across completed Stage 61/62 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(
        path
        for path in args.campaign_root.rglob("*")
        if path.is_file() and path.suffix.lower() != ".pt"
    )
    payload = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v9_stage61_stage62_inventory_v1",
            "status": "completed",
            "files": [
                {
                    "path": path.relative_to(args.campaign_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": compute_file_hash(str(path)),
                }
                for path in files
            ],
            "heavy_checkpoints_omitted": [
                {
                    "path": path.relative_to(args.campaign_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": compute_file_hash(str(path)),
                }
                for path in sorted(args.campaign_root.rglob("*.pt"))
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
