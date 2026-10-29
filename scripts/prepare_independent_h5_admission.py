"""Inventory or sequentially admit the 24 independent TRAIN24 event H5 streams."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "metadata", "admit", "verify"))
    parser.add_argument(
        "--raw-receipt",
        type=Path,
        default=Path("output/independent_phase/raw_history_index_v4/RECEIPT.json"),
    )
    parser.add_argument("--eap-root", type=Path, default=Path("E:/eAP_dataset"))
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=Path("data/manifests/eap_detection_public_files_v1.json"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("output/independent_phase/raw_history_h5_inventory_v2/INVENTORY.json"),
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path(
            "output/independent_phase/raw_history_h5_metadata_preflight_v4/PREFLIGHT.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/independent_phase/raw_history_h5_admission_v1"),
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--full-hash", action="store_true")
    return parser.parse_args()


def _atomic(run: object, output: Path) -> dict[str, object]:
    root = (Path.cwd() / "output" / "independent_phase").resolve()
    destination = output.resolve()
    if destination.parent != root or destination.exists():
        raise ValueError(f"Fresh output must be a direct, nonexistent child of {root}")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{destination.name}.building.{uuid.uuid4().hex}"
    try:
        result = run(staging)  # type: ignore[operator]
        os.replace(staging, destination)
    except Exception as error:
        if staging.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            failed = root / f"{destination.name}.failed.{stamp}.{uuid.uuid4().hex[:8]}"
            shutil.move(str(staging), str(failed))
            (failed / "FAILED.json").write_text(
                json.dumps(
                    {"status": "FAILED_PRESERVED", "error": str(error)},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    return {**result, "output_dir": str(destination)}


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from operational.independent_refit.h5_admission import (
        build_h5_admission,
        build_h5_inventory,
        build_h5_metadata_preflight,
        verify_h5_admission,
        verify_h5_inventory,
        verify_h5_metadata_preflight,
    )

    if args.mode == "inventory":
        output = args.inventory.parent
        result = _atomic(
            lambda staging: build_h5_inventory(
                args.raw_receipt, args.eap_root, staging, args.public_manifest
            ),
            output,
        )
    elif args.mode == "metadata":
        result = _atomic(
            lambda staging: build_h5_metadata_preflight(args.inventory, staging),
            args.preflight.parent,
        )
    elif args.mode == "admit":
        destination = args.output_dir.resolve()
        resume_dir = destination.with_name(f".{destination.name}.resume")
        result = _atomic(
            lambda staging: build_h5_admission(args.preflight, staging, resume_dir=resume_dir),
            args.output_dir,
        )
    else:
        receipt = args.receipt or args.inventory
        if receipt.name == "ADMISSION.json":
            result = verify_h5_admission(receipt, full_hash=args.full_hash)
        elif receipt.name == "PREFLIGHT.json":
            result = verify_h5_metadata_preflight(receipt)
        else:
            result = verify_h5_inventory(receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
