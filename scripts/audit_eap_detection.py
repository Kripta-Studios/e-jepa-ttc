"""Pinned eAP G0 audit and selective metadata/label recovery CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.detection.data_audit import (
    audit_eap,
    download_recovery_plan,
    read_pinned_manifest,
    recovery_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "recover"])
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/eap_detection_public_files_v1.json")
    )
    parser.add_argument(
        "--contract", type=Path, default=Path("configs/detection/eap_contract_v1.json")
    )
    parser.add_argument("--dataset-root", type=Path, help="Actual local eAP root, if accessible")
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument(
        "--verify-present", action="store_true", help="Hash existing referenced media"
    )
    parser.add_argument("--verify-tar-members", action="store_true")
    parser.add_argument("--download", action="store_true", help="recover only: default is dry-run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "recover":
        report = recovery_plan(read_pinned_manifest(args.manifest), args.metadata_root)
        report["dry_run"] = not args.download
        if args.download:
            report["receipts"] = download_recovery_plan(report, args.metadata_root)
        code = 0
    else:
        if args.download:
            parser.error("--download is valid only with recover")
        report = audit_eap(
            metadata_root=args.metadata_root,
            manifest_path=args.manifest,
            contract_path=args.contract,
            dataset_root=args.dataset_root,
            sample_submission=args.sample_submission,
            verify_present=args.verify_present,
            verify_tar_members=args.verify_tar_members,
        )
        code = 0 if report["status"] == "PASS" else 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report.get("status", "DRY_RUN_OR_RECOVERY"),
                "blockers": report.get("blockers", []),
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
