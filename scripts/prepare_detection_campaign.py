"""Build the deterministic D4096 campaign and its frozen selection panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.detection.campaign_data import (
    audit_population,
    build_campaign_manifest,
    build_full_selection_manifest,
    canonical_sha256,
    load_candidate_split,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--pinned-manifest", type=Path)
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-budget", type=int, default=4096)
    parser.add_argument("--selection-budget", type=int, default=1024)
    parser.add_argument("--full-selection-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.full_selection_only:
        value = build_full_selection_manifest(
            data_root=args.data_root, candidate_split_path=args.split
        )
        args.output_dir.mkdir(parents=True)
        path = args.output_dir / "full_selection_tokens.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt = {
            "schema": "eap_detection_full_selection_receipt_v1",
            "token_count": len(value["sample_tokens"]),
            "canonical_json_sha256": canonical_sha256(value),
            "file_sha256": sha256_file(path),
            "labels_read": False,
        }
        (args.output_dir / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt))
        return 0
    if args.pinned_manifest is None or args.pilot_manifest is None:
        parser.error("campaign build requires --pinned-manifest and --pilot-manifest")
    report, records, _ = audit_population(
        data_root=args.data_root,
        candidate_split_path=args.split,
        pinned_manifest_path=args.pinned_manifest,
    )
    split = load_candidate_split(args.split)
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    manifest, coverage = build_campaign_manifest(
        report=report,
        records=records,
        split=split,
        pilot_manifest=pilot,
        train_budget=args.train_budget,
        selection_budget=args.selection_budget,
    )
    args.output_dir.mkdir(parents=True)
    outputs = {
        "population_audit.json": report,
        "d4096_manifest.json": manifest,
        "d4096_coverage.json": coverage,
    }
    for name, value in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    frame_path = args.output_dir / "frame_population.jsonl"
    with frame_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in sorted(records, key=lambda item: str(item["sample_token"])):
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {
        "schema": "eap_detection_campaign_build_receipt_v1",
        "outputs": {
            name: {
                "canonical_json_sha256": canonical_sha256(value),
                "file_sha256": sha256_file(args.output_dir / name),
            }
            for name, value in outputs.items()
        },
        "frame_population": {
            "rows": len(records),
            "file_sha256": sha256_file(frame_path),
            "purpose": "deterministic resampling without repeating the full geometry audit",
        },
        "train_frames": len(manifest["roles"]["train"]["sample_tokens"]),
        "selection_frames": len(manifest["roles"]["selection"]["sample_tokens"]),
        "test_labels_read": False,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
