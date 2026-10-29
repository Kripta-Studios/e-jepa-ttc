"""Freeze the metadata-only independent rotation-0 R_A5 RGB source."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--role-mask", type=Path, required=True)
    parser.add_argument("--p-manifest", type=Path, required=True)
    parser.add_argument("--h-manifest", type=Path, required=True)
    parser.add_argument("--v-manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-role-receipt", type=Path, required=True)
    parser.add_argument("--p-cache-manifest", type=Path)
    parser.add_argument("--p-cache-parity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from operational.independent_refit.producer_rgb_inputs import build_rgb_source_descriptor

    destination = args.output.resolve()
    root = (Path.cwd() / "output" / "independent_phase").resolve()
    if destination.parent != root or destination.exists():
        raise ValueError("Output must be a fresh direct child of output/independent_phase")
    staging = root / f".{destination.name}.building.{uuid.uuid4().hex}"
    try:
        value = build_rgb_source_descriptor(
            population_path=args.population,
            role_mask_path=args.role_mask,
            role_manifests={"P": args.p_manifest, "H": args.h_manifest, "V": args.v_manifest},
            public_manifest_path=args.public_manifest,
            teacher_manifest_path=args.teacher_manifest,
            teacher_role_receipt_path=args.teacher_role_receipt,
            output_dir=staging,
            p_cache_manifest_path=args.p_cache_manifest,
            p_cache_parity_path=args.p_cache_parity,
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            failed = root / f"{destination.name}.failed.{uuid.uuid4().hex[:8]}"
            shutil.move(str(staging), str(failed))
        raise
    print(json.dumps({**value, "output": str(destination)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
