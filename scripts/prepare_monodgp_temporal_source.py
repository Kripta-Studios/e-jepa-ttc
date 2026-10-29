"""Materialize the controlled copied MonoDGP S3 backend source."""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.temporal_source_patch import materialize_temporal_source


def main() -> int:
    """Parse explicit immutable inputs and write a new variant plus receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-preregistration", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--forbidden-destination",
        type=Path,
        default=Path("output/detection_campaign/portable_monodgp_velocity"),
    )
    args = parser.parse_args()
    result = materialize_temporal_source(
        source_root=args.source_root,
        destination=args.destination,
        checkpoint=args.checkpoint,
        runtime_preregistration=args.runtime_preregistration,
        patch_output=args.patch_output,
        manifest_output=args.manifest_output,
        forbidden_destination=args.forbidden_destination,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
