"""Safely ingest a complete RGB return without SSH or active-fit replacement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational.runpod_migration.rgb_return_archive import admit_local_rgb_return


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    default = repository / "artifacts/runpod_returns/irtz8y3sigd621/final_rgb"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Local path or HTTP(S) URL; SSH is unsupported"
    )
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--destination", type=Path, default=default)
    parser.add_argument("--reuse-verification", type=Path)
    parser.add_argument("--rotation", type=int, action="append", choices=(2, 3, 4))
    args = parser.parse_args()
    value = admit_local_rgb_return(
        args.source,
        expected_sha256=args.expected_sha256,
        destination_root=args.destination,
        repository=repository,
        reuse_verification=args.reuse_verification,
        expected_rotations=tuple(args.rotation or (2, 3, 4)),
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
