"""Verify all returned RGB endpoints with the original scientific validators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational.rgb_port.accounting import atomic_write_json
from operational.runpod_migration.consume import verified_cloud_consumption
from operational.runpod_migration.inventory import digest, read
from operational.runpod_migration.rgb_return_archive import verify_extracted_rgb_return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotation", type=int, action="append", choices=(2, 3, 4))
    args = parser.parse_args()
    rotations_requested = tuple(args.rotation or (2, 3, 4))
    if rotations_requested not in {(2, 3, 4), (4,)}:
        raise ValueError("Scientific verification supports full [2,3,4] or individual [4]")
    repository = Path(__file__).resolve().parents[1]
    root = args.archive_root.resolve(strict=True)
    if args.output.resolve().is_relative_to(root):
        raise ValueError("Scientific report must remain outside the immutable archive tree")
    receipt = read(root / "LOCAL_VERIFICATION.json")
    if receipt.get("status") != "PASS" or receipt.get("archive_sha256") != args.archive_sha256:
        raise ValueError("Archive SHA differs from the pinned local download verification")
    verify_extracted_rgb_return(root, args.archive_sha256)
    manifest = read(root / "MANIFEST.json")
    if manifest.get("rotations") != list(rotations_requested):
        raise ValueError("Archive rotations differ from scientific verifier selection")
    producers = [root / f"roots/r{n}_{kind}" for n in rotations_requested for kind in ("a5", "c2f")]
    heads = [root / f"roots/r{n}_downstream" for n in rotations_requested]
    with verified_cloud_consumption(producers, repository, downstream_roots=heads) as verified:
        rotations = {
            str(report["rotation"]): {
                "endpoints": {
                    fit: {**identity, "original_verify_scientific_admission": "PASS"}
                    for fit, identity in report["endpoints"].items()
                }
            }
            for report in verified["rgb_downstream"]
        }
        value = {
            "schema": "ejepa_local_rgb_scientific_reusability_v1",
            "status": "PASS",
            "archive_sha256": args.archive_sha256,
            "archive_manifest_sha256": digest(root / "MANIFEST.json"),
            "scientific_reusability_verified": True,
            "selected_root_path_maps_verified": True,
            "three_root_path_maps_verified": rotations_requested == (2, 3, 4),
            "active_fits_modified": False,
            "active_local_fits_changed": False,
            "original_bytes_changed": False,
            "optimizer_updates": 0,
            "cross_hardware_bit_exact_training_claimed": False,
            "interface": "PHASE17_RAW_RGB",
            "final_ttc_ready": False,
            "verification": verified,
            "rotations": rotations,
        }
    atomic_write_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
